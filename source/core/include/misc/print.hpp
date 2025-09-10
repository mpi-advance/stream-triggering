#ifndef ST_MISC_PRINT
#define ST_MISC_PRINT

#include <iostream>

namespace Print
{
/** @brief set rank to normally impossible value */
inline int rank = -10;

/** @brief set rank to new_rank
 *  @param [in] new_rank rank to set. 
 */
inline void initialize_rank(int new_rank)
{
    Print::rank = new_rank;
}

/** @brief Recursively prints the items supplied in args. 
 *  @details 
 *	
 * 
 *  @param [in] Args
 *  @param [in] args lists of arguments to print 
 */
template <typename T, typename... Args>
void print_out_r(const T& arg, Args&&... args)
{
    std::cout << arg << " ";
    if constexpr (sizeof...(Args))  // If still have other parameters
        print_out_r(std::forward<Args>(args)...);
    else
        std::cout << std::endl;
}

/** @brief Recursively prints the ranks supplied in args. 
 *  @details 
 *	
 * 
 *  @param [in] Args
 *  @param [in] args lists of arguments to print 
 */
template <bool UseRanks = true, typename... Args>
void always(Args&&... args)
{
    if constexpr (UseRanks)
    {
        std::cout << "Rank: " << Print::rank << " - ";
    }
    print_out_r(std::forward<Args>(args)...);
}

template <bool UseRanks = true, typename... Args>
void out(Args&&... args)
{
#ifndef NDEBUG
    always<UseRanks>(std::forward<Args>(args)...);
#endif
}

}  // namespace Print

#endif