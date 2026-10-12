#ifndef ST_MISC_PRINT
#define ST_MISC_PRINT

#include <iostream>

/**
 * @file print.hpp
 * @brief Custom debugging and message printing tools.
 */

/** @brief Custom debugging and message printing tools. */
namespace Print
{
/** @brief Variable all print statements will use when told to print process rank.
 * @details Defaults to -10, and is set by Print::initialize_rank()
 */
inline int rank = -10;

/**
 * @brief Set the rank of this process (for printout purposes).
 * @details Any printout called before this function is called will still work. The
 * printing functions will just use the default value for Print::rank (which is -10).
 * @param [in] new_rank The new rank to use.
 */
inline void initialize_rank(int new_rank)
{
    Print::rank = new_rank;
}

/**
 * @brief Underlying recursive function for printing multiple items.
 * @details This function prints out multiple items, each automatically followed by a
 * space. It uses recursive calls to itself to print through the list of items to print. A
 * new line is printed after everything else is printed. This function is not generally
 * called by users since it has no "if not in debug mode, don't print" logic contained in
 * it. Users should call Print::out(), Print::always(), or Print::alwaysZ().
 *
 * @tparam T The type of the first item to print out
 * @tparam Args The remaining parameter pack of the types of things to print through
 * @param [in] arg The first item to print out
 * @param [in] args The remaining parameter pack of things to print through
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

/**
 * @brief Print something to standard out, regardless of build mode.
 * @details If UseRanks is set, then the string "Rank: <#> - " is printed out before all
 * of the items are printed out via Print::print_out_r().
 *
 * @tparam UseRanks A compile-time option to determine if the printout should be prefaced
 * by a printout of the rank doing the printing.
 * @tparam Args The remaining parameter pack of the types of things to print through
 * @param [in] args The set of items to print out.
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

/**
 * @brief Print something to standard out, regardless of build mode, if process rank 0.
 * @details If the process is not rank 0 (according to Print::rank), nothing happens from
 * calling this function. Otherwise, the function passes everything along to
 * Print::always().
 *
 * @tparam UseRanks A compile-time option to determine if the printout should be prefaced
 * by a printout of the rank doing the printing.
 * @tparam Args The remaining parameter pack of the types of things to print through
 * @param [in] args The set of items to print out.
 */
template <bool UseRanks = true, typename... Args>
void alwaysZ(Args&&... args)
{
    if (Print::rank == 0)
    {
        always<UseRanks>(std::forward<Args>(args)...);
    }
}

/**
 * @brief Print something to standard out.
 * @details A function to print something to standard out in a way that will turn itself
 * off if the library is not built in debug mode (will print if NDEBUG is not defined).
 * Uses a template parameter pack to recursively print things out.
 *
 * @tparam UseRanks A compile-time option to determine if the printout should be prefaced
 * by a printout of the rank doing the printing.
 * @tparam Args The remaining parameter pack of the types of things to print through
 * @param [in] args The set of items to print out.
 */
template <bool UseRanks = true, typename... Args>
void out(Args&&... args)
{
#ifndef NDEBUG
    always<UseRanks>(std::forward<Args>(args)...);
#endif
}

}  // namespace Print

#endif