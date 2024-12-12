from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd


CNM_DECODER = "tensor_gemma_decoder"
CNM_SDPA = "tensor_gemma_sdpa"
CNM_MLP = "tensor_gemma_mlp"
CNM_LINEAR = "tensor_linear"
CNM_RMSNORM = "tensor_rms_norm"
CNM_TRANSP = "tensor_transp"
CNM_ROTARY = "tensor_apply_zamba_rotary_embedding"
CNM_SDP_MASK = "sdp_attention_masked_scaled_fwd"
CNM_MM = "tensor_mm"
CNM_MUL = "tensor_mul"
CNM_ADD = "tensor_add"
CNM_SFX = "tensor_softmax"
CNM_GELU = "tensor_gelu"
CNM_FMLP = "tensor_gemma_fmlp"


def read_data(path: str) -> pd.DataFrame:
    ucols = ['ID', 'Function Name', 'gpc__cycles_elapsed.max [cycle]', 'gpu__time_duration.sum [ms]']
    df = pd.read_csv(path, header=0, usecols=ucols)
    return df


def crt_kernel_map(w_fmlp=False) -> dict:
    kernel_location_map = {
        0: [CNM_DECODER, CNM_RMSNORM],
        1: [CNM_DECODER, CNM_SDPA, CNM_LINEAR],
        2: [CNM_DECODER, CNM_SDPA, CNM_LINEAR],
        3: [CNM_DECODER, CNM_SDPA, CNM_LINEAR],

        4: [CNM_DECODER, CNM_SDPA, CNM_TRANSP],
        5: [CNM_DECODER, CNM_SDPA, CNM_TRANSP],
        6: [CNM_DECODER, CNM_SDPA, CNM_TRANSP],

        7: [CNM_DECODER, CNM_SDPA, CNM_ROTARY],
        8: [CNM_DECODER, CNM_SDPA, CNM_ROTARY],
        9: [CNM_DECODER, CNM_SDPA, CNM_ROTARY]
    }

    # add the sdp attention related data
    num_mtcs = 8
    offset = 9
    for _ in range(num_mtcs):
        kernel_location_map[offset + 1] = [CNM_DECODER, CNM_SDPA, CNM_SDP_MASK, CNM_TRANSP]
        kernel_location_map[offset + 2] = [CNM_DECODER, CNM_SDPA, CNM_SDP_MASK, CNM_MM]
        kernel_location_map[offset + 3] = [CNM_DECODER, CNM_SDPA, CNM_SDP_MASK, CNM_MUL]
        kernel_location_map[offset + 4] = [CNM_DECODER, CNM_SDPA, CNM_SDP_MASK, CNM_ADD]
        kernel_location_map[offset + 5] = [CNM_DECODER, CNM_SDPA, CNM_SDP_MASK, CNM_SFX]
        kernel_location_map[offset + 6] = [CNM_DECODER, CNM_SDPA, CNM_SDP_MASK, CNM_MM]
        offset += 6
    
    # remaining kernels
    kernel_location_map[offset + 1] = [CNM_DECODER, CNM_SDPA, CNM_TRANSP]
    kernel_location_map[offset + 2] = [CNM_DECODER, CNM_SDPA, CNM_MM]

    kernel_location_map[offset + 3] = [CNM_DECODER, CNM_RMSNORM]
    kernel_location_map[offset + 4] = [CNM_DECODER, CNM_ADD]
    kernel_location_map[offset + 5] = [CNM_DECODER, CNM_RMSNORM]

    if not w_fmlp:
        kernel_location_map[offset + 6] = [CNM_DECODER, CNM_MLP, CNM_MM]
        kernel_location_map[offset + 7] = [CNM_DECODER, CNM_MLP, CNM_GELU]
        kernel_location_map[offset + 8] = [CNM_DECODER, CNM_MLP, CNM_MM]
        kernel_location_map[offset + 9] = [CNM_DECODER, CNM_MLP, CNM_MUL]
        kernel_location_map[offset + 10] = [CNM_DECODER, CNM_MLP, CNM_MM]

        kernel_location_map[offset + 11] = [CNM_DECODER, CNM_RMSNORM]
        kernel_location_map[offset + 12] = [CNM_DECODER, CNM_ADD]
    else:
        kernel_location_map[offset + 6] = [CNM_DECODER, CNM_MLP, CNM_FMLP]
        kernel_location_map[offset + 7] = [CNM_DECODER, CNM_MLP, CNM_MM]

        kernel_location_map[offset + 8] = [CNM_DECODER, CNM_RMSNORM]
        kernel_location_map[offset + 9] = [CNM_DECODER, CNM_ADD]
    
    return kernel_location_map


def calc_decoder_total_kernel_time(df: pd.DataFrame) -> float:
    return df['gpu__time_duration.sum [ms]'].sum()


def calc_level2_kernel_times(df: pd.DataFrame, kernel_loc_map: dict):
    l2_kernel_times = defaultdict(float)
    for kernel_id, location in kernel_loc_map.items():
        kernel_group_nm = location[1]
        l2_kernel_times[kernel_group_nm] += df['gpu__time_duration.sum [ms]'].loc[kernel_id]
    return l2_kernel_times


def calc_leaf_kernel_times(df: pd.DataFrame, kernel_loc_map: dict):
    l2_kernel_times = defaultdict(float)
    for kernel_id, location in kernel_loc_map.items():
        kernel_group_nm = location[-1]
        l2_kernel_times[kernel_group_nm] += df['gpu__time_duration.sum [ms]'].loc[kernel_id]
    return l2_kernel_times


def plot_kernel_times(l2_kernel_times: dict):
    plt.bar(list(l2_kernel_times.keys()), list(l2_kernel_times.values()))
    plt.xticks(rotation=45, ha='right')
    plt.show()


if __name__ == '__main__':
    df = read_data("C:\\Data\\AI\\projects\\anyGPU\\artifacts\\performance\\gemma_decoder\\gemma_decoder_fmlp_v2_gpu_profile.csv")

    print(df.head(5))

    print(calc_decoder_total_kernel_time(df))

    kernel_map = crt_kernel_map(True)
    kts = calc_level2_kernel_times(df, kernel_map)
    plot_kernel_times(kts)

    kts = calc_leaf_kernel_times(df, kernel_map)
    plot_kernel_times(kts)

