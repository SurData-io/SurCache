#include "surdata_transfer.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <errno.h>
#include <mutex>
#include <new>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

struct surdata_event {
    std::atomic<int> signaled{0};
    std::atomic<int> refs{1};
    void* cuda_event{nullptr};
};

struct surdata_fence {
    std::atomic<int> signaled{0};
    std::atomic<int> refs{1};
};

static void fence_release(surdata_fence_t* fence) {
    if (!fence) return;
    if (fence->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete fence;
}

static void event_release(surdata_event_t* event) {
    if (!event) return;
    if (event->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete event;
}

namespace {

constexpr uint32_t kJobs = 128;
constexpr uint32_t kSpans = 32768;
constexpr uint32_t kCompletions = 256;
constexpr int kCudaNotReady = 600;
constexpr unsigned kEventDisableTiming = 2;

enum JobState { kEmpty = 0, kQueued = 1, kDone = 2, kSubmitting = 3 };

struct Job {
    surdata_job_id_t id{};
    uint64_t request_epoch = 0;
    uint64_t generation = 0;
    int direction = 0;
    int cancel = 0;
    int on_device = 0;
    JobState state = kEmpty;
    surdata_fence_t* fence = nullptr;
    surdata_event_t* done_event = nullptr;
    uint32_t nspan = 0;
    int has_result = 0;
    int result_status = 0;
    uint64_t result_bytes = 0;
    surdata_copy_span_t* spans = nullptr;
};

Job g_jobs[kJobs];
surdata_completion_t g_done[kCompletions];
uint32_t g_done_r = 0;
uint32_t g_done_w = 0;
std::mutex g_mu;
std::condition_variable g_cv;
std::thread g_h2d;
std::thread g_d2h;
std::atomic<int> g_stop{0};
int g_started = 0;

void* g_cuda = nullptr;
void* g_stream_h2d = nullptr;
void* g_stream_d2h = nullptr;
int (*p_stream_create)(void**) = nullptr;
int (*p_stream_wait)(void*, void*, unsigned) = nullptr;
int (*p_event_create)(void**, unsigned) = nullptr;
int (*p_event_record)(void*, void*) = nullptr;
int (*p_event_query)(void*) = nullptr;
int (*p_memcpy_async)(void*, const void*, size_t, int, void*) = nullptr;
int (*p_host_register)(void*, size_t, unsigned) = nullptr;
int (*p_host_unregister)(void*) = nullptr;
int (*p_malloc)(void**, size_t) = nullptr;
int (*p_free)(void*) = nullptr;
int (*p_stream_sync)(void*) = nullptr;
int (*p_event_destroy)(void*) = nullptr;
int (*p_get_device)(int*) = nullptr;
int (*p_set_device)(int) = nullptr;
int g_cuda_device = 0;
int g_cuda_device_known = 0;
int g_cuda_bind_epoch = 0;
int g_device_lookup_done = 0;
int g_bound_mapped = 0;
std::atomic<int> g_hold_device_issue{0};
thread_local int t_cuda_epoch = 0;

bool same_job(const surdata_job_id_t& a, const surdata_job_id_t& b) {
    return a.engine_epoch == b.engine_epoch && a.worker_epoch == b.worker_epoch &&
           a.sequence == b.sequence;
}

void* open_mapped_cudart() {
    FILE* maps = std::fopen("/proc/self/maps", "r");
    if (!maps) return nullptr;
    char line[4096];
    void* handle = nullptr;
    while (std::fgets(line, sizeof(line), maps) != nullptr) {
        char* path = std::strchr(line, '/');
        if (!path || std::strstr(path, "libcudart.so") == nullptr) continue;
        size_t n = std::strlen(path);
        while (n > 0 && (path[n - 1] == '\n' || path[n - 1] == '\r' || path[n - 1] == ' ')) path[--n] = '\0';
        if (n == 0) continue;
        handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (handle) break;
    }
    std::fclose(maps);
    return handle;
}

void drop_cuda_binding() {
    g_stream_h2d = nullptr;
    g_stream_d2h = nullptr;
    p_stream_create = nullptr;
    p_stream_wait = nullptr;
    p_event_create = nullptr;
    p_event_record = nullptr;
    p_event_query = nullptr;
    p_memcpy_async = nullptr;
    p_host_register = nullptr;
    p_host_unregister = nullptr;
    p_malloc = nullptr;
    p_free = nullptr;
    p_stream_sync = nullptr;
    p_event_destroy = nullptr;
    p_get_device = nullptr;
    p_set_device = nullptr;
    g_cuda_device_known = 0;
    g_device_lookup_done = 0;
    g_cuda_bind_epoch += 1;
}

bool load_cuda() {
    if (!g_bound_mapped) {
        void* mapped = open_mapped_cudart();
        if (mapped) {
            if (g_cuda != mapped) drop_cuda_binding();
            g_cuda = mapped;
            g_bound_mapped = 1;
        }
    }
    if (g_stream_h2d && g_stream_d2h && p_memcpy_async && p_host_register && g_device_lookup_done) return true;
    if (!g_cuda) {
        const char* names[] = {"libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11", nullptr};
        for (int i = 0; names[i] != nullptr; ++i) {
            g_cuda = dlopen(names[i], RTLD_LAZY | RTLD_LOCAL);
            if (g_cuda) break;
        }
    }
    if (!g_cuda) return false;
    if (!p_stream_create)
        p_stream_create = reinterpret_cast<int (*)(void**)>(dlsym(g_cuda, "cudaStreamCreate"));
    if (!p_stream_wait)
        p_stream_wait = reinterpret_cast<int (*)(void*, void*, unsigned)>(dlsym(g_cuda, "cudaStreamWaitEvent"));
    if (!p_event_create)
        p_event_create = reinterpret_cast<int (*)(void**, unsigned)>(dlsym(g_cuda, "cudaEventCreateWithFlags"));
    if (!p_event_record)
        p_event_record = reinterpret_cast<int (*)(void*, void*)>(dlsym(g_cuda, "cudaEventRecord"));
    if (!p_event_query)
        p_event_query = reinterpret_cast<int (*)(void*)>(dlsym(g_cuda, "cudaEventQuery"));
    if (!p_memcpy_async)
        p_memcpy_async = reinterpret_cast<int (*)(void*, const void*, size_t, int, void*)>(dlsym(g_cuda, "cudaMemcpyAsync"));
    if (!p_host_register)
        p_host_register = reinterpret_cast<int (*)(void*, size_t, unsigned)>(dlsym(g_cuda, "cudaHostRegister"));
    if (!p_host_unregister)
        p_host_unregister = reinterpret_cast<int (*)(void*)>(dlsym(g_cuda, "cudaHostUnregister"));
    if (!p_malloc)
        p_malloc = reinterpret_cast<int (*)(void**, size_t)>(dlsym(g_cuda, "cudaMalloc"));
    if (!p_free)
        p_free = reinterpret_cast<int (*)(void*)>(dlsym(g_cuda, "cudaFree"));
    if (!p_stream_sync)
        p_stream_sync = reinterpret_cast<int (*)(void*)>(dlsym(g_cuda, "cudaStreamSynchronize"));
    if (!p_event_destroy)
        p_event_destroy = reinterpret_cast<int (*)(void*)>(dlsym(g_cuda, "cudaEventDestroy"));
    if (!p_get_device)
        p_get_device = reinterpret_cast<int (*)(int*)>(dlsym(g_cuda, "cudaGetDevice"));
    if (!p_set_device)
        p_set_device = reinterpret_cast<int (*)(int)>(dlsym(g_cuda, "cudaSetDevice"));
    g_device_lookup_done = 1;
    if (!p_stream_create || !p_stream_wait || !p_event_create || !p_event_record || !p_event_query ||
        !p_memcpy_async) {
        return false;
    }
    if (!g_stream_h2d && p_stream_create(&g_stream_h2d) != 0) return false;
    if (!g_stream_d2h && p_stream_create(&g_stream_d2h) != 0) return false;
    if (g_stream_h2d && g_stream_d2h && p_get_device && !g_cuda_device_known) {
        int dev = 0;
        if (p_get_device(&dev) == 0) {
            g_cuda_device = dev;
            g_cuda_device_known = 1;
            g_cuda_bind_epoch += 1;
        }
    }
    return g_stream_h2d && g_stream_d2h && p_host_register;
}

int bind_worker_device() {
    if (t_cuda_epoch == g_cuda_bind_epoch && g_cuda_device_known) return 0;
    if (!p_set_device || !g_cuda_device_known) return SURDATA_DONE_DEVICE;
    if (p_set_device(g_cuda_device) != 0) return SURDATA_DONE_DEVICE;
    t_cuda_epoch = g_cuda_bind_epoch;
    return 0;
}

bool producers_ready(const Job& job) {
    if (job.cancel) return true;
    if (job.fence && job.fence->signaled.load(std::memory_order_acquire) == 0) return false;
    for (uint32_t i = 0; i < job.nspan; ++i) {
        surdata_event_t* ev = job.spans[i].producer;
        if (!ev) continue;
        if (ev->cuda_event) return false;
        if (ev->signaled.load(std::memory_order_acquire) == 0) return false;
    }
    return true;
}

bool push_done_locked(const Job& job, int status, uint64_t bytes) {
    uint32_t next = (g_done_w + 1) % kCompletions;
    if (next == g_done_r) return false;
    surdata_completion_t& c = g_done[g_done_w];
    c.job = job.id;
    c.request_epoch = job.request_epoch;
    c.generation = job.generation;
    c.valid_bytes = bytes;
    c.direction = job.direction;
    c.status = status;
    g_done_w = next;
    return true;
}

void release_spans_locked(Job& job) {
    delete[] job.spans;
    job.spans = nullptr;
    job.nspan = 0;
}

void finish_job_locked(Job& job, int status, uint64_t bytes) {
    job.has_result = 1;
    job.result_status = status;
    job.result_bytes = bytes;
    if (!push_done_locked(job, status, bytes)) {
        job.state = kDone;
        return;
    }
    surdata_fence_t* fence = job.fence;
    surdata_event_t* done = job.done_event;
    job.fence = nullptr;
    job.done_event = nullptr;
    job.has_result = 0;
    job.on_device = 0;
    release_spans_locked(job);
    job.state = kEmpty;
    if (fence) fence_release(fence);
    if (done) event_release(done);
}

void run_job(Job& job) {
    if (job.on_device) return;
    if (job.has_result) {
        std::lock_guard<std::mutex> lock(g_mu);
        finish_job_locked(job, job.result_status, job.result_bytes);
        return;
    }
    if (job.cancel) {
        std::lock_guard<std::mutex> lock(g_mu);
        finish_job_locked(job, SURDATA_DONE_CANCELLED, 0);
        return;
    }
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < job.nspan; ++i) {
        surdata_copy_span_t span = job.spans[i];
        if (!span.src || !span.dst || span.bytes == 0) {
            std::lock_guard<std::mutex> lock(g_mu);
            finish_job_locked(job, SURDATA_DONE_INVALID, 0);
            return;
        }
        std::memcpy(span.dst, span.src, static_cast<size_t>(span.bytes));
        bytes += span.bytes;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    finish_job_locked(job, SURDATA_DONE_SUCCESS, bytes);
}

int issue_spans(int direction, const surdata_copy_span_t* spans, uint32_t nspan, surdata_event_t** out_done);

void worker_main(int direction) {
    while (!g_stop.load(std::memory_order_acquire)) {
        Job* host = nullptr;
        surdata_copy_span_t* local = nullptr;
        uint32_t local_n = 0;
        int local_dir = direction;
        surdata_job_id_t local_id{};
        int local_slot = -1;
        bool device = false;
        bool retry_later = false;
        {
            std::unique_lock<std::mutex> lock(g_mu);
            for (uint32_t i = 0; i < kJobs; ++i) {
                Job& job = g_jobs[i];
                if (job.direction != direction || job.state != kDone || !job.has_result) continue;
                finish_job_locked(job, job.result_status, job.result_bytes);
                if (job.state == kDone && job.has_result) retry_later = true;
                break;
            }
            if (!retry_later && g_hold_device_issue.load(std::memory_order_acquire) == 0) {
                for (uint32_t i = 0; i < kJobs; ++i) {
                    Job& job = g_jobs[i];
                    if (job.state != kQueued || job.direction != direction || !job.on_device || job.done_event) {
                        continue;
                    }
                    if (job.cancel) {
                        finish_job_locked(job, SURDATA_DONE_CANCELLED, 0);
                        if (job.state == kDone && job.has_result) retry_later = true;
                        break;
                    }
                    job.state = kSubmitting;
                    local_n = job.nspan;
                    local_dir = job.direction;
                    local_id = job.id;
                    local_slot = static_cast<int>(i);
                    local = new (std::nothrow) surdata_copy_span_t[local_n ? local_n : 1];
                    if (!local || !job.spans || local_n == 0) {
                        delete[] local;
                        local = nullptr;
                        local_slot = -1;
                        finish_job_locked(job, SURDATA_DONE_DEVICE, 0);
                        if (job.state == kDone && job.has_result) retry_later = true;
                        break;
                    }
                    for (uint32_t s = 0; s < local_n; ++s) local[s] = job.spans[s];
                    device = true;
                    break;
                }
            }
            if (!retry_later && !device) {
                for (uint32_t i = 0; i < kJobs; ++i) {
                    Job& job = g_jobs[i];
                    if (job.state != kQueued || job.direction != direction || job.on_device) continue;
                    if (!producers_ready(job)) continue;
                    host = &job;
                    job.state = kDone;
                    break;
                }
            }
            if (!host && !device && !retry_later) {
                g_cv.wait_for(lock, std::chrono::milliseconds(5));
                continue;
            }
        }
        if (device) {
            bool cancelled = false;
            {
                std::lock_guard<std::mutex> lock(g_mu);
                Job& held = g_jobs[local_slot];
                if (held.state == kSubmitting && same_job(held.id, local_id) && held.cancel) {
                    finish_job_locked(held, SURDATA_DONE_CANCELLED, 0);
                    cancelled = true;
                }
            }
            if (cancelled) {
                delete[] local;
                continue;
            }
            int bound = bind_worker_device();
            surdata_event_t* done = nullptr;
            int issued = bound != 0 ? bound : issue_spans(local_dir, local, local_n, &done);
            delete[] local;
            std::lock_guard<std::mutex> lock(g_mu);
            if (local_slot < 0) {
                if (done) event_release(done);
                continue;
            }
            Job& job = g_jobs[static_cast<uint32_t>(local_slot)];
            if (job.state == kEmpty || job.state != kSubmitting || !same_job(job.id, local_id)) {
                if (done) event_release(done);
                continue;
            }
            if (issued != SURDATA_DONE_SUCCESS) {
                if (done) event_release(done);
                finish_job_locked(job, issued, 0);
                continue;
            }
            job.done_event = done;
            job.state = kQueued;
            continue;
        }
        if (retry_later) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        bool republish = host->has_result && host->state == kDone;
        run_job(*host);
        if (republish && host->state == kDone) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

int submit(surdata_job_id_t id, const surdata_copy_span_t* spans, uint32_t nspan,
           uint64_t request_epoch, uint64_t generation, int direction, surdata_fence_t* fence,
           int on_device) {
    if (!g_started || !spans || nspan == 0 || nspan > kSpans) return SURDATA_SUBMIT_INVALID;
    for (uint32_t i = 0; i < nspan; ++i) {
        if (!spans[i].src || !spans[i].dst || spans[i].bytes == 0) return SURDATA_SUBMIT_INVALID;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    for (uint32_t i = 0; i < kJobs; ++i) {
        if (g_jobs[i].state != kEmpty && same_job(g_jobs[i].id, id)) return SURDATA_SUBMIT_INVALID;
    }
    Job* slot = nullptr;
    for (uint32_t i = 0; i < kJobs; ++i) {
        if (g_jobs[i].state == kEmpty) {
            slot = &g_jobs[i];
            break;
        }
    }
    if (!slot) return SURDATA_SUBMIT_QUEUE_FULL;
    surdata_copy_span_t* copy = new (std::nothrow) surdata_copy_span_t[nspan];
    if (!copy) return SURDATA_SUBMIT_QUEUE_FULL;
    for (uint32_t i = 0; i < nspan; ++i) copy[i] = spans[i];
    slot->id = id;
    slot->request_epoch = request_epoch;
    slot->generation = generation;
    slot->direction = direction;
    slot->cancel = 0;
    slot->on_device = on_device;
    slot->has_result = 0;
    slot->result_status = 0;
    slot->result_bytes = 0;
    slot->fence = fence;
    slot->done_event = nullptr;
    if (fence) fence->refs.fetch_add(1, std::memory_order_acq_rel);
    release_spans_locked(*slot);
    slot->nspan = nspan;
    slot->spans = copy;
    slot->state = kQueued;
    g_cv.notify_all();
    return SURDATA_SUBMIT_ACCEPTED;
}

int issue_spans(int direction, const surdata_copy_span_t* spans, uint32_t nspan, surdata_event_t** out_done) {
    if (out_done) *out_done = nullptr;
    void* stream = direction == SURDATA_DIR_H2D ? g_stream_h2d : g_stream_d2h;
    int kind = direction == SURDATA_DIR_H2D ? 1 : 2;
    for (uint32_t i = 0; i < nspan; ++i) {
        surdata_event_t* ev = spans[i].producer;
        if (ev && ev->cuda_event) {
            if (p_stream_wait(stream, ev->cuda_event, 0) != 0) return SURDATA_DONE_DEVICE;
        }
        if (p_memcpy_async(spans[i].dst, spans[i].src, static_cast<size_t>(spans[i].bytes), kind, stream) != 0) {
            return SURDATA_DONE_DEVICE;
        }
    }
    surdata_event_t* done = new surdata_event();
    if (p_event_create(&done->cuda_event, kEventDisableTiming) != 0 ||
        p_event_record(done->cuda_event, stream) != 0) {
        delete done;
        return SURDATA_DONE_DEVICE;
    }
    if (out_done) *out_done = done;
    return SURDATA_DONE_SUCCESS;
}

}  // namespace

int surdata_transfer_init(void) {
    if (!g_started) {
        g_stop.store(0, std::memory_order_release);
        g_done_r = g_done_w = 0;
        for (uint32_t i = 0; i < kJobs; ++i) g_jobs[i] = Job{};
        g_h2d = std::thread(worker_main, SURDATA_DIR_H2D);
        g_d2h = std::thread(worker_main, SURDATA_DIR_D2H);
        g_started = 1;
    }
    // Connector init can run before the device exists. Pin retries init.
    if (!g_stream_h2d || !g_stream_d2h || !p_memcpy_async || !p_host_register) load_cuda();
    return 0;
}

void surdata_transfer_shutdown(void) {
    if (!g_started) return;
    surdata_transfer_drain();
    g_stop.store(1, std::memory_order_release);
    g_cv.notify_all();
    if (g_h2d.joinable()) g_h2d.join();
    if (g_d2h.joinable()) g_d2h.join();
    for (uint32_t i = 0; i < kJobs; ++i) release_spans_locked(g_jobs[i]);
    g_started = 0;
}

surdata_fence_t* surdata_transfer_fence_create(void) {
    return new surdata_fence();
}

void surdata_transfer_fence_signal(surdata_fence_t* fence) {
    if (!fence) return;
    fence->signaled.store(1, std::memory_order_release);
    g_cv.notify_all();
}

void surdata_transfer_fence_destroy(surdata_fence_t* fence) {
    fence_release(fence);
}

surdata_event_t* surdata_transfer_event_create(void) {
    return new surdata_event();
}

int surdata_transfer_event_record(surdata_event_t* event, void* compute_stream) {
    if (!event) return SURDATA_SUBMIT_INVALID;
    if (!compute_stream) {
        event->signaled.store(1, std::memory_order_release);
        g_cv.notify_all();
        return 0;
    }
    load_cuda();
    if (!p_event_record || !p_event_create) return SURDATA_SUBMIT_INVALID;
    if (!event->cuda_event) {
        void* created = nullptr;
        int rc = p_event_create(&created, kEventDisableTiming);
        if (rc != 0 || !created) {
            event->cuda_event = nullptr;
            return rc != 0 ? rc : SURDATA_SUBMIT_INVALID;
        }
        event->cuda_event = created;
    }
    int rc = p_event_record(event->cuda_event, compute_stream);
    if (rc != 0) {
        if (p_event_destroy) p_event_destroy(event->cuda_event);
        event->cuda_event = nullptr;
        return rc;
    }
    g_cv.notify_all();
    return 0;
}

int surdata_transfer_event_query(surdata_event_t* event) {
    if (!event) return 0;
    if (event->cuda_event && p_event_query) {
        int rc = p_event_query(event->cuda_event);
        if (rc == 0) return 1;
        return 0;
    }
    return event->signaled.load(std::memory_order_acquire) ? 1 : 0;
}

void surdata_transfer_event_destroy(surdata_event_t* event) {
    event_release(event);
}

int surdata_transfer_cuda_ready(void) {
    return g_stream_h2d && g_stream_d2h && p_memcpy_async ? 1 : 0;
}

int surdata_transfer_submit_h2d(surdata_job_id_t id, const surdata_copy_span_t* spans,
                                uint32_t nspan, uint64_t request_epoch, uint64_t generation) {
    return submit(id, spans, nspan, request_epoch, generation, SURDATA_DIR_H2D, nullptr, 0);
}

int surdata_transfer_submit_d2h(surdata_job_id_t id, const surdata_copy_span_t* spans,
                                uint32_t nspan, uint64_t request_epoch, uint64_t generation,
                                surdata_fence_t* producer) {
    return submit(id, spans, nspan, request_epoch, generation, SURDATA_DIR_D2H, producer, 0);
}

int surdata_transfer_submit_device(surdata_job_id_t id, const surdata_copy_span_t* spans, uint32_t nspan,
                                   uint64_t request_epoch, uint64_t generation, int direction) {
    if (!surdata_transfer_cuda_ready()) return SURDATA_SUBMIT_INVALID;
    return submit(id, spans, nspan, request_epoch, generation, direction, nullptr, 1);
}

int surdata_transfer_job_has_done_event(surdata_job_id_t id) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (uint32_t i = 0; i < kJobs; ++i) {
        if (g_jobs[i].state == kEmpty || !same_job(g_jobs[i].id, id)) continue;
        return g_jobs[i].done_event && g_jobs[i].done_event->cuda_event ? 1 : 0;
    }
    return -1;
}

void surdata_transfer_debug_hold_device_issue(int hold) {
    g_hold_device_issue.store(hold ? 1 : 0, std::memory_order_release);
    g_cv.notify_all();
}

int surdata_hostmem_alloc(size_t bytes, void** out, size_t* mapped_bytes) {
    if (out) *out = nullptr;
    if (mapped_bytes) *mapped_bytes = 0;
    if (!out || !mapped_bytes || bytes == 0) return SURDATA_SUBMIT_INVALID;
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) page = 4096;
    size_t pagesz = static_cast<size_t>(page);
    size_t rem = bytes % pagesz;
    size_t mapped = bytes;
    if (rem != 0) {
        size_t add = pagesz - rem;
        if (bytes > static_cast<size_t>(-1) - add) return SURDATA_SUBMIT_INVALID;
        mapped = bytes + add;
    }
    void* ptr = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        *out = nullptr;
        return errno ? errno : SURDATA_SUBMIT_INVALID;
    }
    volatile unsigned char* touch = static_cast<volatile unsigned char*>(ptr);
    for (size_t off = 0; off < mapped; off += pagesz) touch[off] = 0;
    if (mlock(ptr, mapped) != 0) {
        int err = errno;
        munmap(ptr, mapped);
        *out = nullptr;
        return err ? err : SURDATA_SUBMIT_INVALID;
    }
    *out = ptr;
    *mapped_bytes = mapped;
    return 0;
}

void surdata_hostmem_free(void* ptr, size_t mapped_bytes) {
    if (!ptr) return;
    if (mapped_bytes == 0) return;
    munlock(ptr, mapped_bytes);
    munmap(ptr, mapped_bytes);
}

int surdata_transfer_host_pin(void* ptr, size_t bytes) {
    if (!ptr || bytes == 0 || !p_host_register) return SURDATA_SUBMIT_INVALID;
    int rc = p_host_register(ptr, bytes, 1u);
    return rc == 0 ? 0 : rc;
}

int surdata_transfer_host_unpin(void* ptr) {
    if (!ptr || !p_host_unregister) return SURDATA_SUBMIT_INVALID;
    int rc = p_host_unregister(ptr);
    return rc == 0 ? 0 : rc;
}

int surdata_transfer_memcpy_async(void* dst, const void* src, size_t bytes, void* stream, int kind) {
    if (!dst || !src || bytes == 0 || !stream) return SURDATA_SUBMIT_INVALID;
    if (kind != SURDATA_DIR_H2D && kind != SURDATA_DIR_D2H) return SURDATA_SUBMIT_INVALID;
    if (!p_memcpy_async && !load_cuda()) return SURDATA_SUBMIT_INVALID;
    if (!p_memcpy_async) return SURDATA_SUBMIT_INVALID;
    int rc = p_memcpy_async(dst, src, bytes, kind, stream);
    return rc == 0 ? 0 : rc;
}

int surdata_transfer_copy_async_check(void* host, size_t bytes, int direction) {
    if (!host || bytes == 0) return SURDATA_SUBMIT_INVALID;
    if (direction != SURDATA_DIR_H2D && direction != SURDATA_DIR_D2H) return SURDATA_SUBMIT_INVALID;
    if (!load_cuda() || !p_malloc || !p_free || !p_stream_sync || !p_memcpy_async || !p_event_create ||
        !p_event_record || !p_event_query || !p_event_destroy) {
        return SURDATA_SUBMIT_INVALID;
    }
    void* stream = direction == SURDATA_DIR_H2D ? g_stream_h2d : g_stream_d2h;
    if (!stream) return SURDATA_SUBMIT_INVALID;
    int kind = direction == SURDATA_DIR_H2D ? 1 : 2;
    const size_t warm_cap = 4096;
    const size_t probe_cap = static_cast<size_t>(256) * 1024 * 1024;
    size_t warm_n = bytes < warm_cap ? bytes : warm_cap;
    void* warm = nullptr;
    if (p_malloc(&warm, warm_n) != 0 || !warm) return SURDATA_SUBMIT_INVALID;
    void* warm_dst = direction == SURDATA_DIR_H2D ? warm : host;
    const void* warm_src = direction == SURDATA_DIR_H2D ? host : static_cast<const void*>(warm);
    int copy_rc = p_memcpy_async(warm_dst, warm_src, warm_n, kind, stream);
    if (copy_rc != 0) {
        p_free(warm);
        return copy_rc;
    }
    int sync_rc = p_stream_sync(stream);
    p_free(warm);
    if (sync_rc != 0) return sync_rc;

    size_t probe_n = bytes < probe_cap ? bytes : probe_cap;
    void* dev = nullptr;
    if (p_malloc(&dev, probe_n) != 0 || !dev) return SURDATA_SUBMIT_INVALID;
    void* dst = direction == SURDATA_DIR_H2D ? dev : host;
    const void* src = direction == SURDATA_DIR_H2D ? host : static_cast<const void*>(dev);
    copy_rc = p_memcpy_async(dst, src, probe_n, kind, stream);
    if (copy_rc != 0) {
        p_stream_sync(stream);
        p_free(dev);
        return copy_rc;
    }
    void* event = nullptr;
    if (p_event_create(&event, kEventDisableTiming) != 0 || !event || p_event_record(event, stream) != 0) {
        if (event) p_event_destroy(event);
        p_stream_sync(stream);
        p_free(dev);
        return SURDATA_DONE_DEVICE;
    }
    int query = p_event_query(event);
    int result = SURDATA_SUBMIT_INVALID;
    if (query == kCudaNotReady) result = 0;
    else if (query == 0) result = SURDATA_PIN_SYNC;
    else result = query;
    sync_rc = p_stream_sync(stream);
    p_event_destroy(event);
    p_free(dev);
    if (sync_rc != 0) return sync_rc;
    return result;
}

uint32_t surdata_transfer_poll(surdata_completion_t* out, uint32_t cap) {
    if (!out || cap == 0) return 0;
    std::lock_guard<std::mutex> lock(g_mu);
    if (p_event_query) {
        for (uint32_t i = 0; i < kJobs; ++i) {
            Job& job = g_jobs[i];
            if (job.state == kSubmitting) continue;
            if (!job.on_device || job.state != kQueued || !job.done_event || !job.done_event->cuda_event) continue;
            int rc = p_event_query(job.done_event->cuda_event);
            if (rc == kCudaNotReady) continue;
            uint64_t bytes = 0;
            for (uint32_t s = 0; s < job.nspan; ++s) bytes += job.spans[s].bytes;
            finish_job_locked(job, rc == 0 ? SURDATA_DONE_SUCCESS : SURDATA_DONE_DEVICE, bytes);
        }
    }
    uint32_t n = 0;
    while (n < cap && g_done_r != g_done_w) {
        out[n++] = g_done[g_done_r];
        g_done_r = (g_done_r + 1) % kCompletions;
    }
    return n;
}

void surdata_transfer_request_cancel(surdata_job_id_t id) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (uint32_t i = 0; i < kJobs; ++i) {
        Job& job = g_jobs[i];
        if (job.state != kQueued && job.state != kSubmitting) continue;
        if (!same_job(job.id, id)) continue;
        job.cancel = 1;
    }
    g_cv.notify_all();
}

void surdata_transfer_drain(void) {
    for (int i = 0; i < 2000; ++i) {
        bool busy = false;
        {
            std::lock_guard<std::mutex> lock(g_mu);
            for (uint32_t j = 0; j < kJobs; ++j) {
                if (g_jobs[j].state != kEmpty) busy = true;
            }
        }
        if (!busy) return;
        g_cv.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
