#ifndef ST_MISC_PRINT
#define ST_MISC_PRINT

#include <iostream>

namespace Print
{
inline int rank = -10;

inline void initialize_rank(int new_rank)
{
    Print::rank = new_rank;
}

template <typename T, typename... Args>
void print_out_r(const T& arg, Args&&... args)
{
    std::cout << arg << " ";
    if constexpr (sizeof...(Args))  // If still have other parameters
        print_out_r(std::forward<Args>(args)...);
    else
        std::cout << std::endl;
}

template <bool UseRanks = true, typename... Args>
void out(Args&&... args)
{
#ifndef NDEBUG
    if constexpr (UseRanks)
    {
        std::cout << "Rank: " << Print::rank << " - ";
    }
    print_out_r(std::forward<Args>(args)...);
#endif
}
}  // namespace Print

#endif