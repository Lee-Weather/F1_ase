"""Replay known walking-to-standing latent pairs without updating the F1 LLC."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from replay_f1 import ROOT, GymReplay, Latents, Video, initial_case, joint_names, load_actor


def interpolate_latents(start, end, alpha):
    """Shortest unit-sphere arc; antipodal skills must not collapse to zero."""
    if alpha <= 0:
        return start.copy()
    if alpha >= 1:
        return end.copy()
    cosine = np.clip(np.dot(start, end), -1., 1.)
    if cosine > .9995:
        value = (1-alpha)*start + alpha*end
    else:
        tangent = end-cosine*start
        if np.linalg.norm(tangent) < 1e-6:
            tangent = np.zeros_like(start)
            tangent[np.argmin(np.abs(start))] = 1
            tangent -= np.dot(tangent, start)*start
        tangent /= np.linalg.norm(tangent)
        angle = np.arccos(cosine)*alpha
        value = np.cos(angle)*start + np.sin(angle)*tangent
    return (value/np.linalg.norm(value)).astype(np.float32)


def summarize(rows, switch_xy, walk_seconds, dt, names):
    """Report stopping only when a low-speed interval lasts at least two seconds."""
    pre = [r for r in rows if walk_seconds-1 < r['time_s'] <= walk_seconds]
    post = [r for r in rows if r['time_s'] > walk_seconds]
    tail = rows[-min(len(rows), round(2/dt)):]
    speeds = lambda group: np.array([np.hypot(r['world_vx'], r['world_vy']) for r in group])
    stable = lambda r: (np.hypot(r['world_vx'], r['world_vy']) < .1
                        and abs(r['yaw_rate']) < .2 and r['tilt_rad'] < .35
                        and not any(x in r['termination'] for x in
                                    ('base_height', 'base_tilt', 'illegal_ground_contact')))
    first_stop, streak = None, 0
    window = round(2/dt)
    for i, row in enumerate(post):
        streak = streak+1 if stable(row) else 0
        if streak >= window:
            first_stop = post[i-window+1]
            break
    final_xy = np.array([rows[-1]['base_x'], rows[-1]['base_y']])
    legs = [n for n in names if any(s in n for s in ('hip', 'knee', 'ankle'))]
    q_range = max(float(np.ptp([r['q_'+n] for r in tail])) for n in legs)
    foot_samples = sum(r['loaded_feet'] for r in tail)
    loaded_speed = (sum(r['loaded_foot_speed_sum'] for r in tail)/foot_samples
                    if foot_samples else None)
    result = dict(pre_switch_mean_speed_m_s=float(speeds(pre).mean()) if pre else None,
                  tail_2s_mean_speed_m_s=float(speeds(tail).mean()),
                  tail_2s_max_speed_m_s=float(speeds(tail).max()),
                  tail_2s_max_abs_yaw_rate=float(max(abs(r['yaw_rate']) for r in tail)),
                  tail_2s_max_tilt_rad=float(max(r['tilt_rad'] for r in tail)),
                  tail_2s_leg_peak_to_peak_rad=q_range,
                  tail_2s_loaded_foot_speed_m_s=loaded_speed,
                  tail_2s_both_feet_loaded_fraction=float(np.mean([r['loaded_feet']==2 for r in tail])),
                  first_sustained_stop_delay_s=(first_stop['time_s']-walk_seconds if first_stop else None),
                  distance_to_sustained_stop_m=(float(np.linalg.norm(
                      np.array([first_stop['base_x'], first_stop['base_y']])-switch_xy))
                      if first_stop and switch_xy is not None else None),
                  post_switch_net_displacement_m=(float(np.linalg.norm(final_xy-switch_xy))
                                                   if switch_xy is not None else None),
                  low_speed_at_end=bool(len(tail)==window and all(stable(r) for r in tail)),
                  quiet_legs_at_end=bool(q_range < .1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--walk-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--stand-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--blend-seconds', type=float, nargs='+', default=[0., .5])
    parser.add_argument('--walk-seconds', type=float, default=3.)
    parser.add_argument('--duration', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--prefix', default='switchstop_')
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    if not (np.isfinite(args.duration) and np.isfinite(args.walk_seconds)
            and 0 < args.walk_seconds < args.duration):
        parser.error('Require 0 < walk-seconds < duration')
    if any(not np.isfinite(b) or b < 0 or b > args.duration-args.walk_seconds for b in args.blend_seconds):
        parser.error('Each blend duration must fit the post-switch interval')
    if args.prefix and Path(args.prefix).name != args.prefix:
        parser.error('prefix must be a filename prefix')
    import isaacgym  # Before torch.
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor = load_actor(args.checkpoint, 'cuda:0')
    if actor.algorithm != 'ASE':
        parser.error('Latent transitions require an ASE checkpoint')
    cfg = yaml.safe_load((ROOT/'ase/data/cfg/humanoid_f1.yaml').read_text())
    cfg['env'].update(actor.control_config)
    nominal_dt = cfg['sim']['dt']*cfg['env']['controlFrequencyInv']
    steps = round(args.duration/nominal_dt)
    switch_step = round(args.walk_seconds/nominal_dt)
    cfg['env']['episodeLength'] = steps+1
    names = joint_names()
    engine = GymReplay(cfg, names, record_video=not args.no_video)
    dt = engine.dt
    task = engine.task
    feet = [task._body_names.index(s+'_ankle_roll_link') for s in ('left', 'right')]
    pairs = list(itertools.product(args.walk_seeds, args.stand_seeds, args.blend_seconds))
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.prefix+'isaac'
    video = Video(args.output/(stem+'_play.mp4')) if not args.no_video else None
    summaries, frames = [], 0
    fields = ['case', 'walk_seed', 'stand_seed', 'blend_s', 'step', 'time_s', 'phase', 'alpha',
              'base_x', 'base_y', 'base_z', 'local_vx', 'local_vy', 'yaw_rate', 'tilt_rad',
              'world_vx', 'world_vy', 'loaded_feet', 'loaded_foot_speed_sum', 'termination',
              *['q_'+n for n in names], *['z_'+str(i) for i in range(32)]]
    try:
        with (args.output/(stem+'_diag.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for case, (walk_seed, stand_seed, blend) in enumerate(pairs, 1):
                start_video = frames/25
                obs = engine.reset(initial_case(0, cfg, names))
                walk, stand = Latents(walk_seed, False).next(), Latents(stand_seed, False).next()
                blend_steps = round(blend/nominal_dt)
                rows, switch_xy = [], None
                for step in range(steps):
                    if step == switch_step:
                        switch_xy = task._humanoid_root_states[0, :2].cpu().numpy().copy()
                    alpha = (0. if step < switch_step else
                             min(1., (step-switch_step+1)/blend_steps) if blend_steps else 1.)
                    phase = 'WALK' if step < switch_step else ('BLEND' if alpha < 1 else 'STAND z')
                    latent = interpolate_latents(walk, stand, alpha)
                    action = actor(obs, torch.as_tensor(latent[None], device=engine.device))
                    obs, values, reasons, q = engine.step(action)
                    if step == steps-1 and not reasons:
                        reasons = ['probe_end']
                    root = task._humanoid_root_states[0].cpu().numpy()
                    foot_v = task._rigid_body_vel[0, feet, :2].cpu().numpy()
                    loaded = task._contact_forces[0, feet, 2].cpu().numpy() > 5.
                    row = dict(case=case, walk_seed=walk_seed, stand_seed=stand_seed, blend_s=blend,
                               step=step+1, time_s=round((step+1)*dt, 6), phase=phase, alpha=alpha,
                               **dict(zip(['base_x', 'base_y', 'base_z', 'local_vx', 'local_vy',
                                           'yaw_rate', 'tilt_rad'], values)),
                               world_vx=float(root[7]), world_vy=float(root[8]),
                               loaded_feet=int(loaded.sum()),
                               loaded_foot_speed_sum=float(np.linalg.norm(foot_v, axis=1)[loaded].sum()),
                               termination='+'.join(reasons))
                    row.update(zip(['q_'+n for n in names], q.tolist()))
                    row.update(zip(['z_'+str(i) for i in range(32)], latent.tolist()))
                    rows.append(row)
                    writer.writerow(row)
                    if video and step % 2 == 0:
                        label = ('F1 ASE epoch{} | case {}/{} | walk z seed {} -> stand z seed {} | blend {:.1f}s\n'
                                 '{:.2f}/{}s | {} | alpha={:.2f} | world speed={:.3f}m/s').format(
                                     actor.epoch, case, len(pairs), walk_seed, stand_seed, blend,
                                     row['time_s'], args.duration, phase, alpha, np.hypot(root[7], root[8]))
                        video.write(engine.render(), label)
                        frames += 1
                    if reasons:
                        break
                metrics = summarize(rows, switch_xy, args.walk_seconds, dt, names)
                completed = len(rows)==steps and not any(r in reasons for r in
                            ('base_height', 'base_tilt', 'illegal_ground_contact'))
                metrics['completed_full_duration'] = completed
                metrics['walking_before_switch'] = bool(metrics['pre_switch_mean_speed_m_s'] is not None
                                                       and metrics['pre_switch_mean_speed_m_s'] > .3)
                metrics['stationary_at_end'] = bool(completed and metrics['walking_before_switch']
                    and metrics['low_speed_at_end'] and metrics['quiet_legs_at_end'])
                summary = dict(case=case, walk_seed=walk_seed, stand_seed=stand_seed, blend_s=blend,
                               seconds=rows[-1]['time_s'], termination=reasons,
                               video_start_s=start_video, video_end_s=frames/25, **metrics)
                summaries.append(summary)
                f.flush()
                print('TRANSITION_CASE '+json.dumps(summary), flush=True)
    finally:
        try:
            if video:
                video.close()
        finally:
            engine.close()
    result = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  epoch=actor.epoch, seed=args.seed, deterministic=True, init='default_zero_velocity',
                  control_config=actor.control_config, dt=dt, walk_seconds=args.walk_seconds,
                  requested_duration_s=args.duration, interpolation='unit sphere shortest arc',
                  diagnostic_criteria=dict(walking_speed_min=.3, stop_speed_max=.1, stop_hold_s=2,
                    stop_abs_yaw_rate_max=.2, stop_tilt_max=.35, tail_leg_peak_to_peak_max=.1),
                  foot_velocity_note='loaded ankle-roll link XY speed proxy, not contact-point slip; Fz > 5N',
                  video_seconds=frames/25, cases=summaries)
    (args.output/(args.prefix+'summary.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (args.output/(args.prefix+'summary.csv')).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print('TRANSITION_DONE '+json.dumps(dict(cases=len(summaries), video_seconds=frames/25)), flush=True)


if __name__ == '__main__':
    main()
