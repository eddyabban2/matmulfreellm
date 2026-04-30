
memory_metrics = [
    "dram__bytes.sum", 
    "dram__bytes.avg", 
    "dram__throughput.sum.pct_of_peak_sustained_elapsed", # throughput as a percentage
    "dram__bytes.sum.per_second"  
    "dram__bytes_read.sum", 
    "dram__bytes_read.avg", 
    "dram__bytes_write.sum", 
    "dram__bytes_write.avg", 

]

time_metrics = [
    "gpu__time_duration.sum"
]
# https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference
stall_metrics = [
    "smsp__pcsamp_warps_issue_stalled_barrier", # barrier stall
    "smsp__pcsamp_warps_issue_stalled_branch_resolving", # branch stall
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall", # dispatch stall 
    "smsp__pcsamp_warps_issue_stalled_drain", # deallocation of memory resources
    # "smsp__pcsamp_warps_issue_stalled_imc_miss", # imc miss (does not apear in our workload?)
    "smsp__pcsamp_warps_issue_stalled_lg_throttle", # l1 instruction cache stall 
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard", # scoreboard stall?
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle", # mat pipeline stall
    "smsp__pcsamp_warps_issue_stalled_membar", # memory barrier
    "smsp__pcsamp_warps_issue_stalled_mio_throttle", # memory input and output stall 
    "smsp__pcsamp_warps_issue_stalled_misc", # misc hardware stall
    "smsp__pcsamp_warps_issue_stalled_no_instructions", # no instructions issued 
    "smsp__pcsamp_warps_issue_stalled_not_selected", # waiting to be selected 
    "smsp__pcsamp_warps_issue_stalled_selected", # instruction issued?
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard", # score board?
    "smsp__pcsamp_warps_issue_stalled_sleeping", # all threads are sleeping
    "smsp__pcsamp_warps_issue_stalled_tex_throttle", # tex stall?
    "smsp__pcsamp_warps_issue_stalled_wait", # fixed latency stall?
    "smsp__pcsamp_warps_issue_stalled_warpgroup_arrive" # war group wait
]
usage_metrics = [
    "sm__cycles_elapsed.avg", # count of all cycles across SMs 
    "sm__cycles_active.avg", # avg active cycles across SMs
    "sm__cycles_elapsed.sum", # count of all cycles across SMs 
    "sm__cycles_active.sum", # avg active cycles across SMs
    
    "smsp__cycles_active.sum",
    "smsp__cycles_active.avg",
    "smsp__cycles_elapsed.sum",
    "smsp__cycles_elapsed.avg",
    "smsp__cycles_elapsed.avg.per_second",
    
    "sm__cycles_active.sum.pct_of_peak_sustained_elapsed", # count of active cycles across all SMs
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed", # avg active cycles across SMs

    "sm__throughput.sum.pct_of_peak_sustained_elapsed", # peak throughoutput percentage
    "sm__throughput.avg.pct_of_peak_sustained_elapsed", # peak throughoutput percentage

    "smsp__throughput.sum.pct_of_peak_sustained_elapsed", # peak throughoutput percentage
    "smsp__throughput.avg.pct_of_peak_sustained_elapsed", # peak throughoutput percentage

    "smsp__throughput.sum.pct_of_peak_sustained_elapsed", # peak throughoutput percentage
    "smsp__throughput.avg.pct_of_peak_sustained_elapsed", # peak throughoutput percentage

    "sm__warps_active.avg", 
    "sm__warps_active.avg"
]


double_precision_metrics = [ "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
       "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
       "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum" ]
single_precision_metrics = ["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
       "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
       "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"]
half_precision_metrics = ["smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
       "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
       "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"]
tensor_core_metrics = ["smsp__ops_path_tensor_op_hmma_pred_on.sum",
       "smsp__ops_path_tensor_op_imma_pred_on.sum"]

double_precision_metrics += ["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_second",
       "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_second",
       "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_second" ]
single_precision_metrics += ["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_second",
       "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_second",
       "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_second" ]
half_precision_metrics += ["smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_second",
       "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_second",
       "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_second"]

tensor_core_metrics += [
    "smsp__inst_executed_pipe_tensor.sum",
    "smsp__inst_executed_pipe_tensor_op_imma.sum",                      
    "smsp__ops_path_tensor_src_bf16_dst_fp32.sum",                                                                                                          
    "smsp__ops_path_tensor_src_fp16_dst_fp16.sum",                                                                                                      
    "smsp__ops_path_tensor_src_fp16_dst_fp32.sum",                                                                                                             
    "smsp__ops_path_tensor_src_tf32_dst_fp32.sum",                                                         
    
    "smsp__inst_executed_pipe_tensor_op_hmma.sum", # half precision matrix multiply count
    "smsp__ops_path_tensor_src_fp16_dst_fp32.sum",  # math ops count
    "smsp__inst_executed_pipe_tensor_op_hmma.sum.per_second", 
    "smsp__ops_path_tensor_src_fp16_dst_fp32.sum.per_second", 
]

blackwell_tensor = [
    "smsp__inst_executed_pipe_tensor_subpipe_hmma.sum.per_second",
    "smsp__inst_executed_pipe_tensor_subpipe_hmma.sum"
]

jetson_memory = [
    "lts__t_sectors_aperture_device_lookup_miss.sum", 
    "lts__t_sectors_aperture_sysmem_lookup_miss.sum", 
    "lts__t_sectors_aperture_peer_lookup_miss.sum",
    "lts__t_sectors.sum", 
    "lts__t_tag_requests_miss.sum", 
    "lts__d_sectors_fill_device.sum", 
    "lts__d_sectors_fill_sysmem.sum", 
    "lts__t_sectors_aperture_device_op_write_lookup_miss.sum", 
    "lts__t_sectors_aperture_sysmem_op_write_lookup_miss.sum",
    "lts__t_sectors_aperture_peer_op_write_lookup_miss.sum" 
]

jetson_stall = [ 
    "smsp__average_warp_latency_issue_stalled_barrier.sum", 
    "smsp__average_warp_latency_issue_stalled_branch_resolving.sum", 
    "smsp__average_warp_latency_issue_stalled_drain.sum", 
    "smsp__average_warp_latency_issue_stalled_imc_miss.sum", 
    "smsp__average_warp_latency_issue_stalled_lg_throttle.sum", 
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.sum", 
    "smsp__average_warp_latency_issue_stalled_long_scoreboard_pipe_l1tex.sum",
    "smsp__average_warp_latency_issue_stalled_math_pipe_throttle.sum", 
    "smsp__average_warp_latency_issue_stalled_membar.sum", 
    "smsp__average_warp_latency_issue_stalled_mio_throttle.sum", 
    "smsp__average_warp_latency_issue_stalled_mio_throttle_pipe_mio.sum", 
    "smsp__average_warp_latency_issue_stalled_misc.sum", 
    "smsp__average_warp_latency_issue_stalled_no_instruction.sum", 
    "smsp__average_warp_latency_issue_stalled_not_selected.sum",
    "smsp__average_warp_latency_issue_stalled_selected.sum", 
    "smsp__average_warp_latency_issue_stalled_short_scoreboard.sum", 
    "smsp__average_warp_latency_issue_stalled_sleeping.sum", 
    "smsp__average_warp_latency_issue_stalled_tex_throttle.sum", 
    "smsp__average_warp_latency_issue_stalled_wait.sum"
]

def all_metrics():
    return ",".join(memory_metrics +
        double_precision_metrics + 
        single_precision_metrics + 
        half_precision_metrics +
        tensor_core_metrics +
        blackwell_tensor +  
        time_metrics + 
        stall_metrics + 
        usage_metrics + 
        jetson_memory)

# python auto_profiler.py -s 100 --max_new_tokens 1 --model_name ridger/MMfreeLM-370M --metrics jetson                                                                                                                                              
def jetson_metrics():
    return ",".join(
       jetson_memory + 
       double_precision_metrics + 
       single_precision_metrics + 
       half_precision_metrics +
       time_metrics + 
       tensor_core_metrics + 
       usage_metrics + 
       jetson_stall
    )


