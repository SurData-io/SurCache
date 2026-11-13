#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    SURDATA_SUBMIT_ACCEPTED = 0,
    SURDATA_SUBMIT_QUEUE_FULL = 1,
    SURDATA_SUBMIT_INVALID = 2
};

enum {
    SURDATA_DONE_SUCCESS = 0,
    SURDATA_DONE_CANCELLED = 1,
    SURDATA_DONE_INVALID = 2,
    SURDATA_DONE_DEVICE = 3
};

enum {
    SURDATA_DIR_H2D = 1,
    SURDATA_DIR_D2H = 2
};

/* copy_async_check: the event was already complete. Not a submit or done code. */
enum {
    SURDATA_PIN_SYNC = 4
};

typedef struct surdata_job_id {
    uint64_t engine_epoch;
    uint64_t worker_epoch;
    uint64_t sequence;
} surdata_job_id_t;

typedef struct surdata_event surdata_event_t;

typedef struct surdata_copy_span {
    void* src;
    void* dst;
    uint64_t bytes;
    uint32_t layer;
    uint32_t reserved;
    /* Per-layer producer. The copy stream waits on this event only.
     * Null means the job-level fence, if any, is the producer. */
    surdata_event_t* producer;
} surdata_copy_span_t;

typedef struct surdata_completion {
    surdata_job_id_t job;
    uint64_t request_epoch;
    uint64_t generation;
    uint64_t valid_bytes;
    int direction;
    int status;
} surdata_completion_t;

typedef struct surdata_fence surdata_fence_t;

int surdata_transfer_init(void);
void surdata_transfer_shutdown(void);

surdata_fence_t* surdata_transfer_fence_create(void);
void surdata_transfer_fence_signal(surdata_fence_t* fence);
void surdata_transfer_fence_destroy(surdata_fence_t* fence);

/* Host record: stream == NULL sets the event and returns 0. A CUDA stream
 * records on that stream and returns the cudaEventRecord code. Non-zero
 * clears cuda_event and leaves the host flag clear. query is cudaEventQuery
 * or the host flag. Neither path calls cudaDeviceSynchronize or
 * cudaStreamSynchronize. */
surdata_event_t* surdata_transfer_event_create(void);
int surdata_transfer_event_record(surdata_event_t* event, void* compute_stream);
int surdata_transfer_event_query(surdata_event_t* event);
void surdata_transfer_event_destroy(surdata_event_t* event);
int surdata_transfer_cuda_ready(void);
/* cudaHostRegister(ptr, bytes, cudaHostRegisterPortable). Empty pointer, zero
 * bytes, or a missing symbol returns SURDATA_SUBMIT_INVALID. */
int surdata_transfer_host_pin(void* ptr, size_t bytes);
int surdata_transfer_host_unpin(void* ptr);

/* Private anonymous mapping. Touches each page, then mlock. Does not load CUDA.
 * On success *out is the base and *mapped_bytes is bytes rounded up to a page.
 * MAP_FAILED or mlock failure leaves *out null and returns an error. */
int surdata_hostmem_alloc(size_t bytes, void** out, size_t* mapped_bytes);
/* munlock then munmap. A null pointer returns immediately. */
void surdata_hostmem_free(void* ptr, size_t mapped_bytes);

/* One warmup copy on the direction's stream, then a probe copy.
 * Returns 0 when the probe event is still running, SURDATA_PIN_SYNC when that
 * event is already complete, and any other code on failure. cudaMalloc failure
 * is SURDATA_SUBMIT_INVALID. Does not call cudaDeviceSynchronize. */
int surdata_transfer_copy_async_check(void* host, size_t bytes, int direction);

/* cudaMemcpyAsync on the caller's stream. kind is SURDATA_DIR_H2D or
 * SURDATA_DIR_D2H. Does not synchronize the stream or the device.
 * Empty pointer, zero bytes, null stream, or any other kind returns
 * SURDATA_SUBMIT_INVALID. */
int surdata_transfer_memcpy_async(void* dst, const void* src, size_t bytes, void* stream, int kind);

int surdata_transfer_submit_h2d(surdata_job_id_t id, const surdata_copy_span_t* spans,
                                uint32_t nspan, uint64_t request_epoch, uint64_t generation);
int surdata_transfer_submit_d2h(surdata_job_id_t id, const surdata_copy_span_t* spans,
                                uint32_t nspan, uint64_t request_epoch, uint64_t generation,
                                surdata_fence_t* producer);
/* CUDA H2D/D2H streams. Queues the job and returns. The device worker issues the
 * spans. Returns INVALID when CUDA is absent, QUEUE_FULL when no slot is free. */
int surdata_transfer_submit_device(surdata_job_id_t id, const surdata_copy_span_t* spans, uint32_t nspan,
                                   uint64_t request_epoch, uint64_t generation, int direction);
/* 1 when this id's slot already has a CUDA done event, 0 when the slot is held
 * without one, -1 when no slot has this id. */
int surdata_transfer_job_has_done_event(surdata_job_id_t id);
/* Non-zero: device jobs stay queued and are not issued. Zero: the device worker
 * issues them. Host memcpy jobs are not held. */
void surdata_transfer_debug_hold_device_issue(int hold);
uint32_t surdata_transfer_poll(surdata_completion_t* out, uint32_t cap);
void surdata_transfer_request_cancel(surdata_job_id_t id);
void surdata_transfer_drain(void);

#ifdef __cplusplus
}
#endif
