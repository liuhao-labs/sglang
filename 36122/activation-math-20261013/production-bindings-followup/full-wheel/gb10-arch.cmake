# Verification configuration only. Keep every target/source/operator, but
# generate common_ops device code for the available GB10 instead of five GPUs.
function(pr36122_common_ops_gb10_arch)
    foreach(target common_ops_sm90_build common_ops_sm100_build)
        get_target_property(options ${target} COMPILE_OPTIONS)
        string(REGEX REPLACE "-gencode=arch=compute_[^,; ]+,code=sm_[^; >]+" "" options "${options}")
        set_property(TARGET ${target} PROPERTY COMPILE_OPTIONS "${options}")
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_121a,code=sm_121a>")
    endforeach()
endfunction()
cmake_language(DEFER CALL pr36122_common_ops_gb10_arch)
