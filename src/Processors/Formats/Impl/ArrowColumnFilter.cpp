#include "ArrowColumnFilter.h"
#define USE_ARROW 1

#if USE_ARROW || USE_ORC || USE_PARQUET
#ifdef ENABLE_QPL_ANALYSIS

#include <base/types.h>
#include <algorithm>
#include <arrow/builder.h>
#include <arrow/array.h>
#include <Common/logger_useful.h>
#include <Columns/ColumnVector.h>
#include <DataTypes/DataTypesNumber.h>
#include <Interpreters/castColumn.h>
#include <Processors/Chunk.h>

// #include <iostream>
// using namespace std;

namespace DB
{

auto maskAfterScan(std::shared_ptr<arrow::Buffer> & buffer, std::vector<uint8_t> & out_mask)
{
    auto scan_operation = qpl::scan_operation::builder(qpl::equals, 1)
            .input_vector_width(8)
            .output_vector_width(1)
            .parser<qpl::parsers::little_endian_packed_array>(buffer->size())
            .is_inclusive(false)
            .build();
    const auto scan_result = qpl::execute<qpl::hardware>(scan_operation,
                                                         buffer->data(),
                                                         buffer->data() + buffer->size(),
                                                         std::begin(out_mask),
                                                         std::end(out_mask));
    return scan_result;

}

template <typename NumericType, typename VectorType = ColumnVector<NumericType>>
static uint32_t execScan(ColumnWithTypeAndName & out_column, uint32_t mask_length,
                std::vector<uint8_t> & mask_after_scan, std::shared_ptr<arrow::Buffer> buffer)
{
    auto & column_data = const_cast<VectorType &>(static_cast<const VectorType &>(*(out_column.column))).getData();
    // auto & column_data = (static_cast<VectorType &>(*(out_column.column))).getData();
    // column_data.reserve(scan_size);
    std::vector<uint8_t> destination(buffer->size(), 0);

    auto select_operation = qpl::select_operation(mask_after_scan.data(), mask_length);
    const auto select_result = qpl::execute<qpl::hardware>(select_operation,
                                                    buffer->data(),
                                                    buffer->data() + buffer->size(),
                                                    destination.begin(),
                                                    destination.end());
    uint32_t filter_num = 0;                                                    
    select_result.handle([&filter_num, &destination, &column_data](uint32_t select_size) -> void {
                                    // Check if everything was alright
                                  filter_num = select_size;
                                  column_data.insert(destination.data(), destination.data() + select_size);
                                },
                                [](uint32_t status_code) -> void {
                                    throw std::runtime_error("Error: Status code - " + std::to_string(status_code));
                                });                                                    
    return filter_num;
}

void ArrowColumnFilter::printComparator()
{
    LOG_WARNING(&Poco::Logger::get("Functions"), "vec_res, initial data: {}", this->comparator_);
}


void ArrowColumnFilter::arrowTableToCHChunk(Chunk & res, std::shared_ptr<arrow::Table> & table)
{
    NameToColumnPtr name_to_column_ptr;
    for (auto column_name : table->ColumnNames())
    {
        std::shared_ptr<arrow::ChunkedArray> arrow_column = table->GetColumnByName(column_name);
        if (!arrow_column)
            throw std::runtime_error("Column is duplicated");

        name_to_column_ptr[std::move(column_name)] = arrow_column;
    }

    // ArrowColumnFilter arrow_column_filter(this->header, DB::comparators::equals, "LO_LINENUMBER");
    arrowColumnsToCHChunkWithFilter(res, name_to_column_ptr);
}

void ArrowColumnFilter::arrowColumnsToCHChunkWithFilter(Chunk & res, NameToColumnPtr & name_to_column_ptr)
{
    Columns columns_list;
    columns_list.reserve(headers.rows());
    std::vector<ColumnWithTypeAndName> column_type_name;
    uint32_t rows = 0;

    for (size_t column_i = 0, columns = headers.columns(); column_i < columns; ++column_i) {
        auto internal_type = std::make_shared<DataTypeUInt8>();
        auto internal_column = internal_type->createColumn();
        column_type_name.push_back({std::move(internal_column), std::move(internal_type), headers.getByPosition(column_i).name});
    }

    std::shared_ptr<arrow::ChunkedArray> filt_arrow_column = name_to_column_ptr[this->filter_column_name];
    for (size_t chunk_i = 0, num_chunks = static_cast<size_t>(filt_arrow_column->num_chunks()); chunk_i < num_chunks; ++chunk_i) {
        std::shared_ptr<arrow::Array> filt_chunk = filt_arrow_column->chunk(chunk_i);
        if (filt_chunk->length() == 0)
            continue;
        std::shared_ptr<arrow::Buffer> filt_buffer = filt_chunk->data()->buffers[1];
        std::vector<uint8_t> mask_after_scan((filt_chunk->length() +7) / 8, 4);
        // cout << "mask vector size: " << mask_after_scan.size() << endl;
        auto scan_result = maskAfterScan(filt_buffer, mask_after_scan);
        uint32_t mask_length = 0;
        // uint32_t scan_size = 0;
        scan_result.handle([&mask_length](uint32_t value) -> void {
                           // Converting total elements processed to the byte size of the mask.
                           mask_length = (value + 7u) / 8u;
                            // while (mask_length)
                            // {
                            //     mask_length &= (mask_length - 1);
                            //     scan_size++;
                            // }
                            mask_length = (value + 7u) / 8u;
                            
                       },
                       [](uint32_t status_code) -> void {
                           throw std::runtime_error("Error: Status code - " + std::to_string(status_code));
                       });

        int after_filt_rows = 0;
        // cout << "total columns: " << headers.columns();
        for (size_t column_i = 0, columns = headers.columns(); column_i < columns; ++column_i) {
            auto arrow_column = name_to_column_ptr[headers.getByPosition(column_i).name]; 
            std::shared_ptr<arrow::Array> chunk = arrow_column->chunk(chunk_i);
            // cout << "chunk data buffers size: " << chunk->data()->buffers.size() << endl;
            std::shared_ptr<arrow::Buffer> buffer = chunk->data()->buffers[1];
            after_filt_rows = execScan<UInt8>(column_type_name[column_i], mask_length, mask_after_scan, buffer);
        }
        rows += after_filt_rows;
    }

    // convert ColumnWithTypeAndName to column_list
    for (size_t column_i = 0, columns = headers.columns(); column_i < columns; ++ column_i) {
        auto column = column_type_name[column_i];
        const ColumnWithTypeAndName & header_column = headers.getByPosition(column_i);
        try
        {
            column.column = castColumn(column, header_column.type);
        }
        catch (Exception & e)
        {
            e.addMessage("while converting column {}" + header_column.name + " from type {} " + column.type->getName() + "to type {}" + header_column.type->getName());
            throw e;
        }
        column.type = header_column.type;
        columns_list.push_back(std::move(column.column));    
    }
    res.setColumns(columns_list, rows);

}
}

#endif
#endif
