if(USE_IMPLEMENTATION STREQUAL NONE)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/ThreadQueue.cc"
	)
elseif(USE_IMPLEMENTATION STREQUAL CUDA)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/CudaQueue.cu"
	)
elseif(USE_IMPLEMENTATION STREQUAL HPE)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/HPEQueue.cc"
	)
endif()