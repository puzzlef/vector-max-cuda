#pragma once
#include <cassert>
#include "_iostream.hxx"




// BUILD
// -----
// Build modes.

#ifndef BUILD_RELEASE
#define BUILD_RELEASE 0
#define BUILD_ERROR   1
#define BUILD_WARNING 2
#define BUILD_INFO    3
#define BUILD_DEBUG   4
#define BUILD_TRACE   5
#endif




// ASSERT
// ------

#ifndef ASSERT
#define ASSERT(exp)           assert(exp)
#define ASSERT_THAT(exp, msg) assert((exp) && (msg))
#endif




// PERFORM
// -------

#ifndef PEFORME
#if !defined(NDEBUG) && defined(BUILD) && BUILD>=BUILD_ERROR
#define PERFORME(...) __VA_ARGS__
#else
#define PERFORME(...)
#endif
#if !defined(NDEBUG) && defined(BUILD) && BUILD>=BUILD_WARNING
#define PERFORMW(...) __VA_ARGS__
#else
#define PERFORMW(...)
#endif
#if !defined(NDEBUG) && defined(BUILD) && BUILD>=BUILD_INFO
#define PERFORMI(...) __VA_ARGS__
#else
#define PERFORMI(...)
#endif
#if !defined(NDEBUG) && defined(BUILD) && BUILD>=BUILD_DEBUG
#define PERFORMD(...) __VA_ARGS__
#else
#define PERFORMD(...)
#endif
#if !defined(NDEBUG) && defined(BUILD) && BUILD>=BUILD_TRACE
#define PERFORMT(...) __VA_ARGS__
#else
#define PERFORMT(...)
#endif
#endif




// PRINT
// -----

#ifndef FPRINTFE
#define FPRINTFE PERFORME(fprintf)
#define FPRINTFW PERFORMW(fprintf)
#define FPRINTFI PERFORMI(fprintf)
#define FPRINTFD PERFORMD(fprintf)
#define FPRINTFT PERFORMT(fprintf)
#endif

#ifndef PRINTFE
#define PRINTFE PERFORME(printf)
#define PRINTFW PERFORMW(printf)
#define PRINTFI PERFORMI(printf)
#define PRINTFD PERFORMD(printf)
#define PRINTFT PERFORMT(printf)
#endif

#ifndef WRITEE
#define WRITEE PERFORME(write)
#define WRITEW PERFORMW(write)
#define WRITEI PERFORMI(write)
#define WRITED PERFORMD(write)
#define WRITET PERFORMT(write)
#endif

#ifndef PRINTE
#define PRINTE PERFORME(print)
#define PRINTW PERFORMW(print)
#define PRINTI PERFORMI(print)
#define PRINTD PERFORMD(print)
#define PRINTT PERFORMT(print)
#endif

#ifndef PRINTLNE
#define PRINTLNE PERFORME(println)
#define PRINTLNW PERFORMW(println)
#define PRINTLNI PERFORMI(println)
#define PRINTLND PERFORMD(println)
#define PRINTLNT PERFORMT(println)
#endif
