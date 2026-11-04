// platform.hpp - small portability macros.
#pragma once

// Promise to the compiler that a pointer is the only route to the memory it
// references within its scope. Lets inner loops auto-vectorise without runtime
// aliasing checks. Both MSVC and GCC/Clang accept `__restrict`.
#define MM_RESTRICT __restrict
