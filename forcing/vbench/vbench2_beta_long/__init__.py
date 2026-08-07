import os
import re
import math
import importlib
import zipfile
from itertools import chain
from pathlib import Path
from vbench.utils import get_prompt_from_filename, save_json, load_json
from vbench2_beta_long.utils import split_video_into_scenes, split_video_into_clips, load_clip_lengths, get_duration_from_json
from vbench2_beta_long.temporal_flickering import filter_static_clips
from vbench import VBench

import subprocess
from .distributed import (
    get_rank,
    barrier,
)

class VBenchLong(VBench):
    def build_full_dimension_list(self, ):
        return ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", 'overall_consistency', "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style", "clip_score"]

    def preprocess(self, videos_path, mode, threshold = 35.0, duration=2, **kwargs):
        if "split_clip" in os.listdir(videos_path):
            # Get all folder names in the split_clip folder
            split_clip_path=os.path.join(videos_path, "split_clip")
            split_clip_folders_count = len([folder for folder in os.listdir(split_clip_path) if os.path.isdir(os.path.join(split_clip_path, folder))])
            
            # Get the number of files in the videos_path folder that end with '.mp4'
            mp4_files_count = len([file for file in os.listdir(videos_path) if file.endswith('.mp4')])
            
            # Check if the number of folders matches the number of .mp4 files
            if split_clip_folders_count == mp4_files_count:
                print(f"Videos have been splitted into clips in {videos_path}/split_clip")
                return 

        split_rank = int(kwargs.get("split_rank", 0))
        split_world_size = int(kwargs.get("split_world_size", 1))

        # split video into clips
        base_output_dir = os.path.join(videos_path, "split_clip")
        os.makedirs(base_output_dir, exist_ok=True)

        def _sort_key(fname):
            # Handle names like "0-0_ema.mp4" or "1-0_ema.mp4"
            stem = os.path.splitext(fname)[0]
            return int(stem.split("-")[0])
        all_video_files = sorted(
            [f for f in os.listdir(videos_path) if f.endswith((".mp4", ".avi", ".mov"))],
            key=_sort_key
        )

        chunk = math.ceil(len(all_video_files) / split_world_size)
        start = split_rank * chunk
        end = min(start + chunk, len(all_video_files))
        video_files = all_video_files[start:end]

        # detect transistions
        split_scene_video_path = []
        if kwargs['use_semantic_splitting']:
            for video_file in video_files:
                video_path = os.path.join(videos_path, video_file)
                if not video_path.endswith(('.mp4', '.avi', '.mov')):
                    continue
                
                # semantically consistent scenes splitting
                video_name = os.path.splitext(video_file)[0]
                output_dir = os.path.join(videos_path, "split_scene", video_name)
                os.makedirs(output_dir, exist_ok=True)
                split_scene_flag = split_video_into_scenes(video_path, output_dir, threshold)
                if split_scene_flag:
                    split_scene_video_path.append(video_path)

        full_info_list = load_json(self.full_info_dir)
        dimension_clip_length_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", kwargs['clip_length_config'])
        dimension_clip_length = load_clip_lengths(dimension_clip_length_config_path)

        for video_file in video_files:
            video_path = os.path.join(videos_path, video_file)

            if not video_path.endswith(('.mp4', '.avi', '.mov')):
                continue

            duration = get_duration_from_json(video_path, full_info_list, dimension_clip_length)
            if mode == 'long_custom_input':
                duration = 2

            if video_path in split_scene_video_path:
                video_name = os.path.splitext(video_file)[0]
                video_scenes_path = os.path.join(os.path.dirname(video_path), "split_scene", video_name)
                for video_scene_path in os.listdir(video_scenes_path):
                    video_scene_path = os.path.join(video_scenes_path, video_scene_path)
                    split_video_into_clips(video_scene_path, base_output_dir, int(duration), fps=8)

            else:
                split_video_into_clips(video_path, base_output_dir, int(duration), fps=8)

        print(f"[rank {split_rank}] Splitting videos into clips in {base_output_dir} done for {len(video_files)} videos.")

        if split_world_size > 1:
            barrier()


    def evaluate(self, videos_path, name, prompt_list, dimension, read_frame=False, mode='long_vbench_standard', **kwargs):
        dimensions = self.build_full_dimension_list()
        is_dimensional_structure = any(os.path.isdir(os.path.join(videos_path, dim)) for dim in dimensions)
        kwargs['preprocess_dimension_flag'] = dimension
        if is_dimensional_structure:
            for dim in dimensions:
                dimension_path = os.path.join(videos_path, dim)
                self.preprocess(dimension_path, mode, **kwargs)
        else:
            self.preprocess(videos_path, mode, **kwargs)

        # long videos have been splitted into clips
        results_dict = {}
        submodules_dict = self.init_submodules(dimension, read_frame)
        # loop for build_full_info_json for clips
        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension, prompt_list, mode=mode, **kwargs)
        
        dimension_module = importlib.import_module(f'vbench2_beta_long.{dimension}')
        evaluate_func = getattr(dimension_module, f'compute_long_{dimension}')
        submodules_list = submodules_dict[dimension]

        results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
        results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_'+dimension+'_eval_results.json')
        save_json(results_dict, output_name)
        print(f'Evaluation results saved to {output_name}')


    def build_full_info_json(self, videos_path, name, dimension, prompt_list=[], special_str='', verbose=False, mode='vbench_standard', **kwargs):
        cur_full_info_list=[]

        if mode=='long_vbench_standard':
            full_info_list = load_json(self.full_info_dir)
            video_names = os.listdir(videos_path)
            postfix = Path(video_names[0]).suffix
            video_clip_folder_names = [name.replace(postfix, '') for name in video_names]
            for prompt_dict in full_info_list:
                # if the prompt belongs to any dimension we want to evaluate
                if set(dimension) & set(prompt_dict["dimension"]):
                    prompt = prompt_dict['prompt_en']
                    prompt_dict['video_list'] = []
                    for i in range(kwargs['num_of_samples_per_prompt']): # video index for the same prompt
                        intended_video_name_floder = f'{prompt}{special_str}-{str(i)}'
                        intended_video_clips_name_floder = os.path.join(videos_path, "split_clip", intended_video_name_floder)

                        if not os.path.exists(intended_video_clips_name_floder):
                            print(f'WARNING!!! This required video clips are not found! Missing benchmark videos can lead to unfair evaluation result. The missing video clips folder is: {intended_video_clips_name_floder}')
                            continue
                        for video_clip_name in os.listdir(intended_video_clips_name_floder):
                            if video_clip_name.split('_')[0] in video_clip_folder_names:
                                intended_video_path = os.path.join(intended_video_clips_name_floder, video_clip_name)
                                prompt_dict['video_list'].append(intended_video_path)
                            if verbose:
                                print(f'Successfully found video clips in : {intended_video_name_floder}')

                    cur_full_info_list.append(prompt_dict)
        elif mode=='long_custom_input':
            cur_full_info_dict = {}

            # get splitted video paths
            splited_videos_path = os.path.join(videos_path, 'split_clip')

            for prompt_folder in os.listdir(splited_videos_path):
                prompt_folder_path = os.path.join(splited_videos_path, prompt_folder)
                if not os.path.isdir(prompt_folder_path):
                    continue  # Skip if it's not a directory
                
                prefix = prompt_folder.split('_')[0]
                idx_str = prefix.split('-')[0]
                idx = int(idx_str)
                base_prompt = prompt_list[idx]

                if base_prompt not in cur_full_info_dict:
                    cur_full_info_dict[base_prompt] = {
                        "prompt_en": base_prompt,
                        "dimension": dimension,
                        "video_list": []
                    }

                for video_file in os.listdir(prompt_folder_path):
                    if video_file.endswith(('.mp4', '.avi', '.mov')):
                        video_path = os.path.join(prompt_folder_path, video_file)
                        cur_full_info_dict[base_prompt]["video_list"].append(video_path)
            cur_full_info_list = list(cur_full_info_dict.values())

        cur_full_info_path = os.path.join(self.output_path, name+'_'+dimension+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path

    def init_submodules(self, dimension, read_frame=False):
        # dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")
        submodules_dict = {}

        CACHE_DIR = os.environ.get('VBENCH_CACHE_DIR')
        assert CACHE_DIR is not None, "CACHE_DIR is not set"
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        if get_rank() > 0:
            barrier()

        if dimension == 'background_consistency':
            vit_path = f'{CACHE_DIR}/clip_model/ViT-B-32.pt'
            if not os.path.isfile(vit_path):
                wget_command = ['wget', 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', '-P', os.path.dirname(vit_path)]
                subprocess.run(wget_command, check=True)
            dreamsim_path = f'{CACHE_DIR}/dreamsim_model'
            os.makedirs(dreamsim_path, exist_ok=True)
            dreamsim_ensemble_path = os.path.join(dreamsim_path, 'dino_vitb16_pretrain.pth')
            if not os.path.isfile(dreamsim_ensemble_path):
                zip_path = os.path.join(dreamsim_path, 'pretrained.zip')
                wget_command = ['wget', 'https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip', '-O', zip_path]
                unzip_command = ['unzip', '-d', dreamsim_path, zip_path]
                remove_command = ['rm', '-r', zip_path]
                subprocess.run(wget_command, check=True)
                subprocess.run(unzip_command, check=True)
                subprocess.run(remove_command, check=True)
            submodules_dict[dimension] = [dreamsim_path, read_frame]
        elif dimension == 'human_action':
            umt_path = f'{CACHE_DIR}/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth'
            if not os.path.isfile(umt_path):
                wget_command = ['wget', 'https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth', '-P', os.path.dirname(umt_path)]
                subprocess.run(wget_command, check=True)
            submodules_dict[dimension] = [umt_path,]
        elif dimension == 'temporal_flickering':
            submodules_dict[dimension] = []
        elif dimension == 'motion_smoothness':
            CUR_DIR = os.path.dirname(os.path.abspath(__file__))
            submodules_dict[dimension] = {
                    'config': f'{CUR_DIR}/third_party/amt/cfgs/AMT-S.yaml',
                    'ckpt': f'{CACHE_DIR}/amt_model/amt-s.pth'
                }
            details = submodules_dict[dimension]
            if not os.path.isfile(details['ckpt']):
                wget_command = ['wget', 'https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth', '-P', os.path.dirname(details['ckpt'])]
                subprocess.run(wget_command, check=True)
        elif dimension == 'dynamic_degree':
            submodules_dict[dimension] = {
                'model': f'{CACHE_DIR}/raft_model/models/raft-things.pth'
            }
            details = submodules_dict[dimension]
            if not os.path.isfile(details['model']):
                wget_command = ['wget', 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip', '-P', f'{CACHE_DIR}/raft_model/']
                unzip_command = ['unzip', '-d', f'{CACHE_DIR}/raft_model/', f'{CACHE_DIR}/raft_model/models.zip']
                remove_command = ['rm', '-r', f'{CACHE_DIR}/raft_model/models.zip']
                subprocess.run(wget_command, check=True)
                subprocess.run(unzip_command, check=True)
                subprocess.run(remove_command, check=True)
        elif dimension == 'subject_consistency':
            submodules_dict[dimension] = {
                'repo_or_dir': f'{CACHE_DIR}/dino_model/facebookresearch_dino_main/',
                'path': f'{CACHE_DIR}/dino_model/dino_vitbase16_pretrain.pth', 
                'model': 'dino_vitb16',
                'source': 'local',
                'read_frame': read_frame
                }
            details = submodules_dict[dimension]
            if not os.path.isdir(details['repo_or_dir']):
                subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/dino', details['repo_or_dir']], check=True)
            if not os.path.isfile(details['path']):
                wget_command = ['wget', '-P', os.path.dirname(details['path']),
                                'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth']
                subprocess.run(wget_command, check=True)
        elif dimension == 'aesthetic_quality':
            aes_path = f'{CACHE_DIR}/aesthetic_model/emb_reader'
            vit_l_path = f'{CACHE_DIR}/clip_model/ViT-L-14.pt'
            if not os.path.isfile(vit_l_path):
                wget_command = ['wget' ,'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt', '-P', os.path.dirname(vit_l_path)]
                subprocess.run(wget_command, check=True)
            submodules_dict[dimension] = [vit_l_path, aes_path]
        elif dimension == 'imaging_quality':
            musiq_spaq_path = f'{CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
            if not os.path.isfile(musiq_spaq_path):
                wget_command = ['wget', 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth', '-P', os.path.dirname(musiq_spaq_path)]
                subprocess.run(wget_command, check=True)
            submodules_dict[dimension] = {'model_path': musiq_spaq_path}
        elif dimension in ["object_class", "multiple_objects", "color", "spatial_relationship" ]:
            submodules_dict[dimension] = {
                "model_weight": f'{CACHE_DIR}/grit_model/grit_b_densecap_objectdet.pth'
            }
            if not os.path.exists(submodules_dict[dimension]['model_weight']):
                wget_command = ['wget', 'https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth', '-P', os.path.dirname(submodules_dict[dimension]["model_weight"])]
                subprocess.run(wget_command, check=True)
        elif dimension == 'scene':
            submodules_dict[dimension] = {
                "pretrained": f'{CACHE_DIR}/caption_model/tag2text_swin_14m.pth',
                "image_size":384, 
                "vit":"swin_b"
            }
            if not os.path.exists(submodules_dict[dimension]['pretrained']):
                wget_command = ['wget', 'https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth', '-P', os.path.dirname(submodules_dict[dimension]["pretrained"])]
                subprocess.run(wget_command, check=True)
        elif dimension == 'appearance_style':
            submodules_dict[dimension] = {"name": f'{CACHE_DIR}/clip_model/ViT-B-32.pt'}
            if not os.path.isfile(submodules_dict[dimension]["name"]):
                wget_command = ['wget', 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', '-P', os.path.dirname(submodules_dict[dimension]["name"])]
                subprocess.run(wget_command, check=True)
        elif dimension in ["temporal_style", "overall_consistency"]:
            submodules_dict[dimension] = {
                "pretrain": f'{CACHE_DIR}/ViCLIP/ViClip-InternVid-10M-FLT.pth',
            }
            if not os.path.exists(submodules_dict[dimension]['pretrain']):
                wget_command = ['wget', 'https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth', '-P', os.path.dirname(submodules_dict[dimension]["pretrain"])]
                subprocess.run(wget_command, check=True)
        elif dimension == 'clip_score':
            vit_path = f'{CACHE_DIR}/clip_model/ViT-B-32.pt'
            if not os.path.isfile(vit_path):
                wget_command = ['wget', 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', '-P', os.path.dirname(vit_path)]
                subprocess.run(wget_command, check=True)    
            submodules_dict[dimension] = [vit_path]
        if get_rank() == 0:
            barrier()
            
        return submodules_dict
