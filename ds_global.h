#ifndef DS_GLOBAL_H
#define DS_GLOBAL_H

#include <cstdint>

            /* This file would be useful in case I need to
             * use PETSc data types instead of stdint or normal data types in future
             */

typedef uint_fast64_t Index;        // more expressive name for an Index into an array
typedef uint_fast64_t Count;        // more expressive name for an integer quantity
typedef Index NodeId;               // node indices are 0-based
typedef double EdgeWeight;          // edge weight type
typedef Index EdgeId;               // edge id
typedef double Volume;              // Volume, future volume have decimal places
typedef int SmallSize;              // I don't think I need a very large number
typedef uint_fast64_t LargeSize;    // For large volume of objects, I can use this one






#endif // DS_GLOBAL_H



            /* I defined some of the types from below url (Jun 12, 2015 - 12:00)
             * http://en.cppreference.com/w/cpp/types/integer
             */
