if(USE_IMPLEMENTATION STREQUAL NONE)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/ThreadQueue.cc"
	)
elseif(USE_IMPLEMENTATION STREQUAL GPU_MEM_OPS AND
       GPU_TYPE STREQUAL NVIDIA)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/CudaQueue.cpp"
			"${CMAKE_CURRENT_SOURCE_DIR}/CudaQueue.cu"
	)
elseif(USE_IMPLEMENTATION STREQUAL GPU_MEM_OPS AND
	   GPU_TYPE STREQUAL AMD)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/HIPQueue.cc"
 )
elseif(USE_IMPLEMENTATION STREQUAL HPE)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/HPEQueue.cc"
	)
elseif(USE_IMPLEMENTATION STREQUAL CXI)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/CXIQueue.cpp"
	)
endif()
