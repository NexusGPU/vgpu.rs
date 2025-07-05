/* DO NOT MODIFY THIS MANUALLY! This file was generated using cbindgen.
 *
 * This file is generated based on the configuration in
 * `rust/src/build.rs`
 */


#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

void set_limit(uint32_t gpu, uint32_t up_limit, uint64_t mem_limit);

extern int tf_health_check(void);

extern int tf_suspend(void);

extern int tf_resume(void);

extern int tf_vram_reclaim(void);
