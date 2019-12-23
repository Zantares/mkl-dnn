/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "mkldnn.h"
#include "mkldnn_thread.hpp"

#if MKLDNN_THR == MKLDNN_THR_TBB \
                || MKLDNN_THR == MKLDNN_THR_EIGEN \
                || MKLDNN_THR == MKLDNN_THR_TENSORFLOW

#include <sys/syscall.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include <mutex>
#include <thread>

namespace mkldnn {
namespace impl {

static int get_nthr() {
#if MKLDNN_THR != MKLDNN_THR_TENSORFLOW
    static int nthr = 0;
    static std::once_flag initialized;
    std::call_once(initialized, [&]{
        nthr = (int)std::thread::hardware_concurrency();
        const char *ont = getenv("OMP_NUM_THREADS");
        if (ont) nthr = atoi(ont);
        if (nthr < 1) nthr = 1;
    });
    return nthr;
#else
    return mkldnn::impl::eigenTp().NumThreads();
#endif
}

static void pin_thread(int tid) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(tid, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    printf("@@@ MKLDNN_DEBUG: pinning thread %d (LWP %d) to CPU %d\n",
            tid, (int)syscall(SYS_gettid), tid);
    fflush(0);
};

static void maybe_pin_threads(const char *envvar) {
    const char *etp = getenv(envvar);
    if (etp && etp[0] != '1')
        return;

    printf("@@@ MKLDNN_DEBUG: pinning threads...\n");

    const int nthr = get_nthr();
#if MKLDNN_THR == MKLDNN_THR_TENSORFLOW || MKLDNN_THR == MKLDNN_THR_EIGEN
    Eigen::Barrier b(nthr);
    auto pinner = [&b](int thr) {
        b.Notify();
        b.Wait();
        pin_thread(mkldnn_get_thread_num());
    };
    thr_ns::parallel_for(0, nthr, pinner);
#else
    tbb::parallel_for(0, nthr, [&](int ithr) { sleep(2); pin_thread(ithr); },
            tbb::static_partitioner());
#endif
}

#if MKLDNN_THR == MKLDNN_THR_EIGEN || MKLDNN_THR == MKLDNN_THR_TENSORFLOW
static extern_thread_pool_t *eigenTp_;

#if MKLDNN_THR == MKLDNN_THR_EIGEN
static void affinity_reset() {
    // a workaround for IntelCaffe that sets sched mask
    // for the master thread, which makes the whole Eigen
    // thread pool belong to core #0
    const int nthr = get_nthr();
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < nthr; ++i)
        CPU_SET(i, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
}
#endif

extern_thread_pool_t &eigenTp() {
#if MKLDNN_THR == MKLDNN_THR_EIGEN
    if (eigenTp_ == nullptr) {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lock(mtx);

        volatile static bool initialized = false;
        if (!initialized) {
            const int nthr = get_nthr();
            affinity_reset();

            eigenTp_ = new Eigen::ThreadPool(nthr);

            // a workaround to pin threads in Eigen thread pool to the cores
            maybe_pin_threads("PIN_EIGEN_THREADS");

            initialized = true;
        }
    }
#endif
    return *eigenTp_;
}

void parallel_for(int start, int end, std::function<void(int)> f) {
    if (end - start == 1) {
        f(start);
        return;
    }

    int nthr = get_nthr();
    auto& tp = mkldnn::impl::eigenTp();

    Eigen::Barrier b(end - start);
    for (int i = start; i < end; ++i)
        tp.ScheduleWithHint([i, &b, &f]{ f(i); b.Notify(); }, i, i + 1);
    b.Wait();
}
#elif MKLDNN_THR == MKLDNN_THR_TBB
void tbb_init() {
    static std::once_flag initialized;
    std::call_once(initialized, []{ maybe_pin_threads("PIN_TBB_THREADS"); });
}
#endif

}
}

#endif // MKLDNN_THR == MKLDNN_THR_TBB
    // || MKLDNN_THR == MKLDNN_THR_EIGEN
    // || MKLDNN_THR == MKLDNN_THR_TENSORFLOW

mkldnn_status_t mkldnn_set_tensorflow_thread_pool(void *tp) {
#if MKLDNN_THR == MKLDNN_THR_TENSORFLOW
    bool reset_affinity = (tp != mkldnn::impl::eigenTp_);
    mkldnn::impl::eigenTp_ = (extern_thread_pool_t *)tp;
    if (reset_affinity)
        mkldnn::impl::maybe_pin_threads("PIN_TF_THREADS");
    printf("@@@ MKLDNN_DEBUG: will use threadpool %p with %d threads\n",
        tp, mkldnn::impl::eigenTp_->NumThreads());
#endif
    return mkldnn::impl::status::success;
}
