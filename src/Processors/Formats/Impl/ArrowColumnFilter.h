#pragma once

#include "config_formats.h"

#if USE_ARROW || USE_ORC || USE_PARQUET
#ifdef ENABLE_QPL_ANALYSIS

#include <DataTypes/IDataType.h>
#include <Core/ColumnWithTypeAndName.h>
#include <Core/Block.h>
#include <arrow/table.h>

#include <qpl/qpl.hpp>

namespace DB
{

class Block;
class Chunk;

enum comparators {
    less,         /**< Represents < */
    greater,      /**< Represents > */
    equals,       /**< Represents == */
    not_equals    /**< Represents != */
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"
class ArrowColumnFilter
{
public:
    using NameToColumnPtr = std::unordered_map<std::string, std::shared_ptr<arrow::ChunkedArray>>;

    ArrowColumnFilter(const Block & header_, comparators comparator, const String & column_name)
        : headers(header_), comparator_(comparator), filter_column_name(column_name) {}

    void arrowColumnsToCHChunkWithFilter(Chunk & res, NameToColumnPtr & name_to_column_ptr);
    void arrowTableToCHChunk(Chunk & res, std::shared_ptr<arrow::Table> & table);
    void printComparator();

private:
    const Block & headers;
    comparators comparator_;
    const String filter_column_name;
    // std::shared_ptr<arrow::ChunkedArray> filt_column;
};
}
#endif
#endif
