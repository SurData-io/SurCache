#include "surdata_transfer.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static int fail(const char* what) {
    std::fprintf(stderr, "FAIL %s\n", what);
    return 1;
}

static int poll_one(surdata_completion_t* out, int spins) {
    for (int i = 0; i < spins; ++i) {
        if (surdata_transfer_poll(out, 1) == 1) return 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return 0;
}

int main() {
    if (surdata_transfer_init() != 0) return fail("init");

    std::vector<unsigned char> hbm(64, 0);
    std::vector<unsigned char> dram(64, 0);
    std::vector<unsigned char> back(64, 0);
    for (int i = 0; i < 64; ++i) hbm[static_cast<size_t>(i)] = static_cast<unsigned char>(i + 1);

    surdata_fence_t* fence = surdata_transfer_fence_create();
    surdata_copy_span_t d2h{};
    d2h.src = hbm.data();
    d2h.dst = dram.data();
    d2h.bytes = hbm.size();
    surdata_job_id_t save{1, 1, 1};
    int rc = surdata_transfer_submit_d2h(save, &d2h, 1, 9, 3, fence);
    if (rc != SURDATA_SUBMIT_ACCEPTED) return fail("submit d2h");

    surdata_completion_t got{};
    for (int i = 0; i < 5; ++i) {
        if (surdata_transfer_poll(&got, 1) != 0) return fail("poll joined before fence");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (dram[0] != 0) return fail("d2h copied before producer fence");

    std::vector<unsigned char> other_src(4, 7);
    std::vector<unsigned char> other_dst(4, 0);
    surdata_copy_span_t h2d_other{};
    h2d_other.src = other_src.data();
    h2d_other.dst = other_dst.data();
    h2d_other.bytes = 4;
    surdata_job_id_t other{1, 1, 2};
    if (surdata_transfer_submit_h2d(other, &h2d_other, 1, 9, 1) != SURDATA_SUBMIT_ACCEPTED) {
        return fail("submit other");
    }
    if (!poll_one(&got, 200)) return fail("other h2d");
    if (got.job.sequence != 2 || got.status != SURDATA_DONE_SUCCESS || other_dst[0] != 7) {
        return fail("other result");
    }
    if (dram[0] != 0) return fail("d2h still waiting moved");
    std::printf("OK   unsignaled D2H stayed queued across polls while another H2D finished\n");

    surdata_transfer_fence_signal(fence);
    if (!poll_one(&got, 200)) return fail("d2h after fence");
    if (got.direction != SURDATA_DIR_D2H || got.status != SURDATA_DONE_SUCCESS || got.valid_bytes != 64) {
        return fail("d2h completion");
    }
    if (std::memcmp(dram.data(), hbm.data(), 64) != 0) return fail("dram bytes");

    surdata_copy_span_t h2d{};
    h2d.src = dram.data();
    h2d.dst = back.data();
    h2d.bytes = 64;
    surdata_job_id_t load{1, 1, 3};
    if (surdata_transfer_submit_h2d(load, &h2d, 1, 9, 3) != SURDATA_SUBMIT_ACCEPTED) return fail("h2d");
    if (!poll_one(&got, 200)) return fail("h2d poll");
    if (std::memcmp(back.data(), hbm.data(), 64) != 0) return fail("hbm roundtrip");
    std::printf("OK   HBM-DRAM-HBM 64 bytes match\n");

    surdata_event_t* ev = surdata_transfer_event_create();
    if (surdata_transfer_event_query(ev) != 0) return fail("event query before record");
    surdata_copy_span_t layers[2]{};
    unsigned char src_a[8], src_b[8], dst_a[8], dst_b[8];
    for (int i = 0; i < 8; ++i) {
        src_a[i] = static_cast<unsigned char>(10 + i);
        src_b[i] = static_cast<unsigned char>(40 + i);
        dst_a[i] = 0;
        dst_b[i] = 0;
    }
    layers[0].src = src_a;
    layers[0].dst = dst_a;
    layers[0].bytes = 8;
    layers[0].layer = 0;
    layers[0].producer = ev;
    layers[1].src = src_b;
    layers[1].dst = dst_b;
    layers[1].bytes = 8;
    layers[1].layer = 1;
    layers[1].producer = ev;
    surdata_job_id_t layered{1, 1, 4};
    if (surdata_transfer_submit_d2h(layered, layers, 2, 9, 4, nullptr) != SURDATA_SUBMIT_ACCEPTED) {
        return fail("submit layers");
    }
    if (surdata_transfer_poll(&got, 1) != 0) return fail("layers ran before producer event");
    if (surdata_transfer_event_record(ev, nullptr) != 0) return fail("event record");
    if (surdata_transfer_event_query(ev) != 1) return fail("event record");
    if (!poll_one(&got, 200)) return fail("layers after record");
    if (got.valid_bytes != 16 || std::memcmp(dst_a, src_a, 8) != 0 || std::memcmp(dst_b, src_b, 8) != 0) {
        return fail("per-layer bytes");
    }
    std::printf("OK   two layer spans waited on one producer event\n");
    int cuda_ready = surdata_transfer_cuda_ready();
    std::printf("OK   cuda_ready=%d\n", cuda_ready);
    if (cuda_ready) {
        surdata_transfer_debug_hold_device_issue(1);
        unsigned char dev_src[8];
        unsigned char dev_dst[8];
        for (int i = 0; i < 8; ++i) {
            dev_src[i] = static_cast<unsigned char>(i + 1);
            dev_dst[i] = 0;
        }
        surdata_copy_span_t dev_span{};
        dev_span.src = dev_src;
        dev_span.dst = dev_dst;
        dev_span.bytes = 8;
        surdata_job_id_t dev_job{1, 1, 90};
        int dev_rc = surdata_transfer_submit_device(dev_job, &dev_span, 1, 9, 1, SURDATA_DIR_H2D);
        int has_done = surdata_transfer_job_has_done_event(dev_job);
        surdata_transfer_debug_hold_device_issue(0);
        if (dev_rc != SURDATA_SUBMIT_ACCEPTED) return fail("device submit");
        if (has_done != 0) return fail("device done event already set");
        if (!poll_one(&got, 200) || got.job.sequence != 90) return fail("device completion");
        std::printf("OK   device submit returns before the done event\n");
        surdata_transfer_debug_hold_device_issue(1);
        surdata_copy_span_t cancel_span{};
        cancel_span.src = dev_src;
        cancel_span.dst = dev_dst;
        cancel_span.bytes = 8;
        surdata_job_id_t cancel_job{1, 1, 91};
        int cancel_rc = surdata_transfer_submit_device(cancel_job, &cancel_span, 1, 9, 1, SURDATA_DIR_H2D);
        surdata_transfer_request_cancel(cancel_job);
        int cancel_has = surdata_transfer_job_has_done_event(cancel_job);
        surdata_transfer_debug_hold_device_issue(0);
        if (cancel_rc != SURDATA_SUBMIT_ACCEPTED) return fail("cancel submit");
        if (cancel_has != 0) return fail("cancelled job already issued");
        if (!poll_one(&got, 200) || got.job.sequence != 91 || got.status != SURDATA_DONE_CANCELLED) {
            return fail("cancelled device completion");
        }
        std::printf("OK   held device job cancels before issue\n");
    } else {
        std::printf("SKIP device submit, cuda_ready=0\n");
    }
    if (surdata_transfer_host_pin(nullptr, 0) == 0) return fail("pin null");
    if (surdata_transfer_host_unpin(nullptr) == 0) return fail("unpin null");
    void* page = nullptr;
    size_t mapped = 0;
    if (surdata_hostmem_alloc(4096, &page, &mapped) != 0 || page == nullptr || mapped < 4096) {
        return fail("hostmem alloc");
    }
    surdata_hostmem_free(page, mapped);
    surdata_hostmem_free(nullptr, mapped);
    if (surdata_transfer_copy_async_check(nullptr, 0, SURDATA_DIR_D2H) == 0) return fail("async check null");
    if (surdata_transfer_memcpy_async(nullptr, nullptr, 0, nullptr, 0) != SURDATA_SUBMIT_INVALID) {
        return fail("memcpy empty");
    }
    int stack_dst = 0;
    int stack_src = 0;
    if (surdata_transfer_memcpy_async(&stack_dst, &stack_src, sizeof(stack_dst), nullptr, SURDATA_DIR_D2H) !=
        SURDATA_SUBMIT_INVALID) {
        return fail("memcpy null stream");
    }
    if (surdata_transfer_memcpy_async(&stack_dst, &stack_src, sizeof(stack_dst), &stack_dst, 0) !=
        SURDATA_SUBMIT_INVALID) {
        return fail("memcpy bad kind");
    }
    if (surdata_transfer_memcpy_async(&stack_dst, &stack_src, 0, &stack_dst, SURDATA_DIR_H2D) !=
        SURDATA_SUBMIT_INVALID) {
        return fail("memcpy zero bytes");
    }
    surdata_transfer_event_destroy(ev);
    surdata_transfer_fence_destroy(fence);
    surdata_transfer_shutdown();
    std::printf("PASS transfer\n");
    return 0;
}
