"""CPU-side recipe/attestation tests; actual CUDA evidence remains mandatory."""
import io

import pytest
from PIL import Image as PILImage
from pydantic import ValidationError

from sls_bench.config import Image
from sls_bench.image_diffusers import guidance_arguments
from sls_bench.image_gate import ValidationFailed, identity, validate_png
from sls_bench.image_matrix import current_generation_profiles, klein_profile
from tests.test_image_gate import check, fixture, png


def rgba():
    output = io.BytesIO()
    PILImage.open(io.BytesIO(png())).convert('RGBA').save(output, format='PNG')
    return output.getvalue()


def test_current_recipes_are_distinct_native_stock_no_offload_40_call_curves():
    flux, qwen = current_generation_profiles()
    assert flux['model'] == 'black-forest-labs/FLUX.2-dev'
    assert flux['steps'] == 50 and flux['width'] == 1024 and flux['min_vram_gb'] == 128
    assert flux['readiness'] == 'ready' and flux['gated_model']
    assert qwen['model'] == 'Qwen/Qwen-Image-2.1'
    assert qwen['pipeline'] == 'QwenImage21Pipeline'
    assert qwen['steps'] == 40 and qwen['width'] == 2048 and qwen['min_vram_gb'] == 48
    assert qwen['guidance_mode'] == 'none' and qwen['true_cfg_scale'] == 1
    assert qwen['use_kv_cache'] is True
    for p in (flux, qwen):
        assert p['requests'] == p['minimum_requests'] == 40
        assert p['warmup_requests'] == 3 and p['execution_modes'] == ['batch']
        assert p['precision'] == p['vae_dtype'] == 'bf16' and p['tier'] == 'stock'
        assert p['offload_policy'] == 'none' and p['extended_curve_budget']
        assert p['requested_denoiser_steps'] == p['steps']
    assert identity(qwen) != identity({**qwen, 'use_kv_cache': False})


def test_klein_is_a_separate_four_step_bf16_recipe_without_offload_or_qwen_cache():
    p = klein_profile()
    assert p['model'] == 'black-forest-labs/FLUX.2-klein-4B'
    assert p['pipeline'] == 'Flux2KleinPipeline' and p['steps'] == p['requested_denoiser_steps'] == 4
    assert p['precision'] == p['vae_dtype'] == 'bf16' and p['tier'] == 'stock'
    assert p['offload_policy'] == 'none' and p['use_kv_cache'] is None
    assert p['width'] == p['height'] == 1024 and p['min_vram_gb'] == 20
    assert p['requests'] == p['minimum_requests'] == 40 and p['warmup_requests'] == 3
    assert guidance_arguments(p) == {'guidance_scale': 1.0}


def test_pipeline_specific_guidance_does_not_pass_flux_arguments_to_qwen21():
    flux, qwen = current_generation_profiles()
    assert guidance_arguments(flux) == {'guidance_scale': 4.0}
    assert guidance_arguments(qwen) == {'true_cfg_scale': 1.0}
    assert guidance_arguments({**qwen, 'guidance_mode': 'true_cfg', 'true_cfg_scale': 4}) == {'true_cfg_scale': 4, 'negative_prompt': ' '}


@pytest.mark.parametrize('pipeline', ['FluxPipeline', 'Flux2Pipeline', 'ZImagePipeline', 'StableDiffusionPipeline', 'StableDiffusionXLPipeline', 'QwenImagePipeline', 'QwenImageEditPipeline'])
@pytest.mark.parametrize('mode', ['none', 'distilled', 'true_cfg'])
def test_existing_pipeline_guidance_contract_is_unchanged(pipeline, mode):
    config = dict(pipeline=pipeline, guidance_mode=mode, guidance_scale=3.5, true_cfg_scale=4.0)
    old_arguments = ({'true_cfg_scale': 4.0, 'negative_prompt': ' '}
                     if mode == 'true_cfg' and pipeline.startswith('Qwen') else {'guidance_scale': 3.5})
    assert guidance_arguments(config) == old_arguments


@pytest.mark.parametrize('pipeline', ['Flux2Pipeline', 'ZImagePipeline', 'StableDiffusionPipeline', 'StableDiffusionXLPipeline', 'QwenImagePipeline', 'QwenImageEditPipeline'])
def test_explicit_prefix_cache_cannot_leak_to_other_models(pipeline):
    with pytest.raises(ValidationError, match='only for QwenImage21Pipeline'):
        Image(**{**current_generation_profiles()[1], 'pipeline': pipeline})


@pytest.mark.parametrize('change', [{'requests': 8, 'minimum_requests': 8}, {'execution_modes': ['latency']}, {'minimum_requests': 8}])
def test_extended_curves_do_not_enable_downsampled_or_unlabelled_long_runs(change):
    with pytest.raises(ValidationError):
        Image(**{**current_generation_profiles()[0], **change})


@pytest.mark.parametrize('pipeline,cfg', [('Flux2Pipeline', False), ('Flux2KleinPipeline', False), ('QwenImage21Pipeline', False), ('QwenImage21Pipeline', True)])
def test_gate_attests_current_architectures_including_rgba_and_true_batch(pipeline, cfg):
    report, config = fixture()
    config.update(pipeline=pipeline, vae_dtype='bf16', guidance_mode='true_cfg' if cfg else 'none', guidance_scale=0, true_cfg_scale=4 if cfg else 1, use_kv_cache=True if pipeline.startswith('Qwen') else None)
    meta = report['image_runs']['1']['metadata']
    meta.update({k: config[k] for k in ('guidance_mode', 'guidance_scale', 'true_cfg_scale', 'use_kv_cache')})
    meta['scheduler'] = 'FlowMatchEulerDiscreteScheduler'
    meta['components']['transformer'] = meta['components'].pop('unet')
    meta['components']['vae']['dtypes'] = ['torch.bfloat16']
    for trial in report['trials']:
        for call in trial['requests']:
            call['vae_decode_input_dtypes'] = ['torch.bfloat16']
            call['denoiser_forward_count'] = 60 if cfg else 30
            call['denoiser_forwards_per_step'] = [2 if cfg else 1] * 30
            call['actual_denoiser_batch_dimensions'] = [trial['batch_size']] * (60 if cfg else 30)
    image = rgba() if pipeline.startswith('Qwen') else png()
    assert check(report, config, lambda *a: image)[0]['b2'] == 'completed'
    report['trials'][1]['requests'][0]['actual_denoiser_batch_dimensions'][0] = 1
    with pytest.raises(ValidationFailed, match='batch dimensions'):
        check(report, config, lambda *a: image)


def test_rgba_is_only_accepted_explicitly_and_rgb_cannot_masquerade_as_rgba():
    validate_png(rgba(), 4, 4, channels=4)
    for data, channels in [(rgba(), 3), (png(), 4)]:
        with pytest.raises(ValidationFailed):
            validate_png(data, 4, 4, channels=channels)
