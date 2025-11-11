#ifndef PMU_WRAPPER_H
#define PMU_WRAPPER_H

#include <stdint.h>

typedef struct {
    uint64_t cycles;
    uint64_t instructions;
    uint64_t l1d_cache_refills;
    uint64_t l2d_cache_refills;
    uint64_t branch_mispredictions;
} PMUCounters;

uint64_t read_cycle_counter(void);
uint64_t read_event_counter_0(void);
uint64_t read_event_counter_1(void);
uint64_t read_event_counter_2(void);
uint64_t read_event_counter_3(void);
void read_all_counters(PMUCounters *counters);

#endif
