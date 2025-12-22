import os
import json
import sys
import signal
import shutil
import yaml
from datetime import datetime
import importlib

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation import Simulation, signal_handler
from core.environment import StreetViewEnvironment
from core.agent import StreetViewAgent

from core.utils import (
    setup_logging_tee,
    close_log_file,
    load_checkpoint_and_prepare,
    generate_run_folder,
    load_paths,
    load_prompts
)

interrupted = False

def _run_single_simulation(
    cfg,
    prompt,
    experiment_folder,
    path,
    is_resume=False,
    resume_folder=None,
    resume_from_decision="latest",
    resume_into_new_folder=True,
):
    log_filepath = os.path.join(experiment_folder, f"terminal_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    os.makedirs(experiment_folder, exist_ok=True)
    setup_logging_tee(log_filepath)

    if is_resume and resume_folder:
        # Determine which checkpoint to load
        checkpoint_path = None
        try:
            entries = os.listdir(resume_folder)
        except Exception:
            entries = []

        decision_numbers = []
        for name in entries:
            if name.startswith("checkpoint_decision_") and name.endswith(".pkl"):
                middle = name[len("checkpoint_decision_"):-len(".pkl")]
                if middle.isdigit():
                    decision_numbers.append(int(middle))

        if resume_from_decision == "latest":
            if decision_numbers:
                chosen = max(decision_numbers)
            else:
                raise FileNotFoundError(f"No decision checkpoints found in {resume_folder}")
        else:
            # Allow int or numeric string
            if isinstance(resume_from_decision, str) and resume_from_decision.isdigit():
                chosen = int(resume_from_decision)
            elif isinstance(resume_from_decision, int):
                chosen = resume_from_decision
            else:
                raise ValueError("resume_from_decision must be an int or 'latest'")

        checkpoint_path = os.path.join(resume_folder, f"checkpoint_decision_{chosen}.pkl")

        # If continuing in the same folder, ensure experiment_folder points to resume_folder
        target_folder = resume_folder if not resume_into_new_folder else experiment_folder

        agent, env, strategy, config, resume_action, loaded_coords = load_checkpoint_and_prepare(
            checkpoint_path,
            target_folder,
            is_branching=bool(resume_into_new_folder),
        )
        # Apply current config prompt toggles and self-positioning setting on resume
        try:
            strategy.prompt_components = cfg.get('prompt_components', {})
            if 'use_self_positioning' in cfg:
                strategy.use_self_positioning = cfg.get('use_self_positioning', strategy.use_self_positioning)
        except Exception:
            pass
        is_resuming = True
    else:
        # Determine initialization mode and parameters
        init_from_pano_id = cfg.get('init_from_pano_id', False)
        initial_pano_id = None
        initial_coords = (path['start']['position']['lat'], path['start']['position']['lng'])

        if init_from_pano_id:
            if 'startPanoId' not in path['start']:
                raise ValueError("init_from_pano_id is True but startPanoId not found in path data.")
            initial_pano_id = path['start']['startPanoId']
        elif 'startPanoId' in path['start']:
            # If startPanoId exists but init_from_pano_id is False, use coordinate initialization
            pass  # Use coordinates as before

        # Get destination polygon if available
        destination_polygon = None
        if 'destinationPolygon' in path.get('end', {}):
            destination_polygon = path['end']['destinationPolygon']

        env = StreetViewEnvironment(
            initial_coords=initial_coords,
            destination_coords=(path['end']['position']['lat'], path['end']['position']['lng']),
            api_key=cfg['maps_api_key'],
            enable_evaluations=cfg.get('enable_evaluations', True),
            init_from_pano_id=init_from_pano_id,
            initial_pano_id=initial_pano_id,
            destination_polygon=destination_polygon
        )
        env.experiment_folder = experiment_folder

        # Select strategy module/class from config
        strategy_module_name = (
            cfg.get('strategy_module')
            or ('strategies.baseline' if cfg.get('strategy_mode') == 'baseline' else 'strategies.memory_strategy')
        )
        strategy_module = importlib.import_module(strategy_module_name)
        StrategyClass = getattr(strategy_module, 'AdvancedStreetViewStrategy')

        strategy = StrategyClass(
            maps_api_key=cfg['maps_api_key'],
            call_folder=experiment_folder,
            navigation_prompt=prompt,
            model_name=cfg['model_name'],
            use_self_positioning=cfg.get('use_self_positioning', True),
            use_memory_manager=cfg.get('use_memory_manager', False),
            llm_params=cfg.get('llm_params'),
            prompt_components=cfg.get('prompt_components'),
            streetview_signing_secret=cfg.get('streetview_signing_secret'),
            use_signed_streetview=cfg.get('use_signed_streetview', True),
        )

        # Enforce presence of end.destination and use it as the destination name
        if 'destination' not in path.get('end', {}):
            raise ValueError("Path is missing end.destination; each path must include an 'end.destination' name.")
        agent = StreetViewAgent(
            strategy_instance=strategy,
            destination=path['end']['destination'],
            memory=""
        )

        initial_obs = env.get_observation()
        agent.update(None, initial_obs)

        config = {
            'initial_coords': (path['start']['position']['lat'], path['start']['position']['lng']),
            'destination_coords': (path['end']['position']['lat'], path['end']['position']['lng']),
            'maps_api_key': cfg['maps_api_key'],
            'destination_name': path['end']['destination'],
            'destination_radius': path['end'].get('destinationRadius', 50),
            'model_name': cfg['model_name'],
            'use_self_positioning': cfg.get('use_self_positioning', True),
            'navigation_prompt': prompt,
            'max_decision_points': cfg.get('max_decision_points', 50),
            'max_steps': cfg.get('max_steps', 700),
            'enable_evaluations': cfg.get('enable_evaluations', True),
            'min_score': cfg.get('min_score', None),
            'termination_criteria': cfg.get('termination_criteria', 'distance'),
            'llm_params': cfg.get('llm_params', {})
        }

        # Get polygon data if available
        polygon_data = None
        if 'destinationPolygon' in path.get('end', {}):
            polygon_data = path['end']['destinationPolygon']

        points_data = {
            'start': {'lat': config['initial_coords'][0], 'lng': config['initial_coords'][1]},
            'end': {'lat': config['destination_coords'][0], 'lng': config['destination_coords'][1]},
            'destinationPolygon': polygon_data,
            'termination_criteria': config.get('termination_criteria', 'distance')
        }
        points_filepath = os.path.join(experiment_folder, 'points.json')
        with open(points_filepath, 'w') as f:
            json.dump(points_data, f, indent=4)

        is_resuming = False
        resume_action = None
        loaded_coords = []

    simulation = Simulation(agent, env, strategy, config, experiment_folder, is_resuming=is_resuming, resume_action=resume_action, initial_visited_coordinates=loaded_coords)

    simulation.run()
    close_log_file()

    if hasattr(env, 'cleanup'):
        env.cleanup()

def run_simulation(config_file='config.yml'):
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    # Fall back to environment variables for API keys if not in config
    if not cfg.get('maps_api_key'):
        cfg['maps_api_key'] = os.environ.get('GOOGLE_MAPS_API_KEY', '')
    if not cfg.get('streetview_signing_secret'):
        cfg['streetview_signing_secret'] = os.environ.get('STREETVIEW_SIGNING_SECRET', '')

    paths = load_paths(cfg['paths_file'])
    prompts = load_prompts(cfg['prompts_file'])
    prompt = prompts[0]

    # Resolve resume config
    is_resume_flag = bool(cfg.get('is_resume', False) or cfg.get('resume_folder'))
    resume_folder = cfg.get('resume_folder')
    resume_from_decision = cfg.get('resume_from_decision', 'latest')
    resume_into_new_folder = cfg.get('resume_into_new_folder', True)

    if cfg.get('batch_mode', False):
        for path in paths:
            for run_idx in range(cfg.get('runs_per_path', 1)):
                # Determine target experiment folder based on resume mode
                if is_resume_flag and resume_folder and not resume_into_new_folder:
                    experiment_folder = resume_folder
                else:
                    experiment_folder = generate_run_folder(path, cfg['model_name'], False, cfg)
                os.makedirs(experiment_folder, exist_ok=True)
                shutil.copy(config_file, experiment_folder)
                _run_single_simulation(
                    cfg,
                    prompt,
                    experiment_folder,
                    path,
                    is_resume=is_resume_flag,
                    resume_folder=resume_folder,
                    resume_from_decision=resume_from_decision,
                    resume_into_new_folder=resume_into_new_folder,
                )
    else:
        path = paths[0]
        if is_resume_flag and resume_folder and not resume_into_new_folder:
            experiment_folder = resume_folder
        else:
            experiment_folder = generate_run_folder(path, cfg['model_name'], False, cfg)
        os.makedirs(experiment_folder, exist_ok=True)
        shutil.copy(config_file, experiment_folder)
        _run_single_simulation(
            cfg,
            prompt,
            experiment_folder,
            path,
            is_resume=is_resume_flag,
            resume_folder=resume_folder,
            resume_from_decision=resume_from_decision,
            resume_into_new_folder=resume_into_new_folder,
        )


def main(config_file='config.yml'):
    run_simulation(config_file)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
