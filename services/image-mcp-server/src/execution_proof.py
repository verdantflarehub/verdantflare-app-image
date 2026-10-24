"""Bounded, read-only evidence for completed ChatGPT image executions."""
import hashlib


def proof(task, artifacts):
    p = task.request_params
    if (task.status != 'completed' or not task.artifact_id or task.engine != 'codex'
            or p.get('channel_runtime') != 'chatgpt-channel-v1'
            or task.model != 'gpt-image-2.5-sunburst'):
        raise ValueError('execution_proof_not_ready')
    prompt = p.get('prompt')
    if not isinstance(prompt, str) or not prompt:
        raise ValueError('execution_proof_not_ready')
    action = 'edit' if p.get('action') == 'inpaint' else p.get('action')
    if action not in {'generate', 'edit'}:
        raise ValueError('execution_proof_not_ready')
    def media(artifact_id):
        a = artifacts.get(artifact_id, task.project_id)
        if a.project_id != task.project_id:
            raise ValueError('execution_proof_project_mismatch')
        return {'sha256': a.sha256, 'size': a.size_bytes, 'mime': a.media_type}
    ids = [p.get('source_artifact_id'), *(p.get('reference_artifact_ids') or []),
           *(p.get('reference_images') or [])]
    ids = list(dict.fromkeys(x for x in ids if x))
    if p.get('mask_artifact_id'):
        ids.append(p['mask_artifact_id'])
    size = {'16:9': '2048x1152', '9:16': '1152x2048', '1:1': '1024x1024',
            '4:3': '1792x1344', '3:4': '1344x1792'}.get(p.get('aspect_ratio'), p.get('size') or '2048x1152')
    parameters = {'action': action, 'size': size,
                  'prefer_4k': (p.get('resolution') or '2k').lower() == '4k',
                  'quality': p.get('quality') or ('high' if p.get('action') == 'generate' else 'auto'),
                  'background': p.get('background') or 'auto'}
    if p.get('instructions') is not None:
        parameters['instructions'] = p['instructions']
    return {'schema_version': 1, 'service_id': 'image', 'run_id': task.task_id,
            'project_id': task.project_id, 'state': 'succeeded', 'model_id': task.model,
            'route': 'codex', 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'parameters': parameters, 'inputs': [media(x) for x in ids],
            'outputs': [media(task.artifact_id)]}
