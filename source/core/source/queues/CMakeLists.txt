if(USE_CUDA_BACKEND)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/CudaQueue.cpp"
			"${CMAKE_CURRENT_SOURCE_DIR}/CudaQueue.cu"
	)
endif()

if(USE_HIP_BACKEND)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/HIPQueue.cc"
 )
endif()

if(USE_CXI_BACKEND)
	target_sources(stream-triggering
		PRIVATE
			"${CMAKE_CURRENT_SOURCE_DIR}/CXIQueue.cpp"
	)
endif()
