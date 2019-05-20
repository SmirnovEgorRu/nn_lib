#!/bin/bash

SOURCES="main.cpp"
NAME="nn"
CC="icc"

#-DTBB_USE_THREADING_TOOLS

TBBROOT="/export/users/cache/tbb/lnx/2017.2.132/tbb"
DAALROOT="/nfs/inn/proj/numerics1/Users/egorsmir/daal/daal_versions/daal_2019u4/__release_lnx/daal/lib/intel64_lin/"
OPTIONS="-g -O3 -Werror -std=c++17"
if [ "${1}" == "skx" ];then
    OPTIONS="${OPTIONS} -xCOMMON-AVX512 -DAVX512"
else # AVX-2
    OPTIONS="${OPTIONS} -xCORE-AVX2"
fi

# INCLUDES=""
# LINK=""
INCLUDES="${TBBROOT}/include/tbb -I include/"
LINK="-tbb -mkl -ltbbmalloc -lpthread ${DAALROOT}/libdaal_core.so ${DAALROOT}/libdaal_thread.so"

echo "${CC} ${OPTIONS} ${SOURCES} -I ${INCLUDES} ${LINK} -DFP_TYPE=float -fopenmp -o ${NAME}.out"
${CC} ${OPTIONS} ${SOURCES} -I ${INCLUDES} ${LINK} -DFP_TYPE=float  -fopenmp -o ${NAME}.out

if [ "${?}" -eq "0" ]; then
    echo "STATUS: SUCCESS"
else
    echo "STATUS: FAILED"
fi
# gcc -E file.cpp -o file.ii