if (ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif ()

if (ENABLE_FUZZING)
    add_subdirectory(fuzzers)
endif()

if (ENABLE_QPL)

    macro (dbms_target_link_libraries)
        foreach (module ${all_modules})
            target_link_libraries (${module} ${ARGN})
        endforeach ()
    endmacro ()
    
    macro (dbms_target_include_directories)
        foreach (module ${all_modules})
            target_include_directories (${module} ${ARGN})
        endforeach ()
    endmacro ()
    dbms_target_link_libraries(PUBLIC ch_contrib::qpl)
    dbms_target_include_directories(SYSTEM BEFORE PUBLIC ${ClickHouse_SOURCE_DIR}/contrib/qpl/include)
endif()
