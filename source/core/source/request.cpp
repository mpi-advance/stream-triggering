#include "abstract/request.hpp"

namespace Communication
{
bool IPC_PROTOCOL_ENABLED    = true;
bool CREDIT_PROTOCOL_ENABLED = true;

MPI_Comm MPIS_COMM_WORLD = MPI_COMM_NULL;

}  // namespace Communication