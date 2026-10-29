"""Read-only, opt-in macro-step visualization; no observations or scans are built here."""
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge

from utils.utils import get_cell_position_from_coords, make_gif


class V1EpisodeGif:
    def __init__(self, root, episode, worker_id, frame_every=1, filename_prefix=''):
        if frame_every < 1:
            raise ValueError('GIF frame interval must be positive')
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=f'episode_{episode:06d}_worker_{worker_id}_', dir=root))
        self.filename_prefix = filename_prefix
        self.episode = episode
        self.worker_id = worker_id
        self.frame_every = frame_every
        self.trajectories = []
        self.frame_files = []
        self.frames = []
        self.render_seconds = 0.0

    def record(self, worker, macro_step, final=False):
        started = time.perf_counter()
        # Copy every executed macro endpoint, even when this step is not rendered.
        self.trajectories.append(np.array(worker.env.robot_locations, copy=True))
        if macro_step != 0 and macro_step % self.frame_every and not final:
            return
        figure, metadata = self.render_frame(worker, macro_step)
        filename = self.directory / f'frame_{macro_step:04d}.png'
        try:
            figure.savefig(filename, dpi=110)
        finally:
            figure.clear()
        self.frame_files.append(str(filename))
        self.frames.append(metadata)
        self.render_seconds += time.perf_counter() - started

    def render_frame(self, worker, macro_step):
        env = worker.env
        info = env.belief_info
        positions = np.asarray(env.robot_locations)
        cells = get_cell_position_from_coords(positions, info).reshape(-1, 2)
        # The same pre-existing shared Belief cache used by the graph; never
        # extract frontiers from GT or refresh the environment for a picture.
        frontiers = np.asarray(list(worker.node_manager.global_frontiers)).reshape(-1, 2)
        frontier_cells = get_cell_position_from_coords(frontiers, info).reshape(-1, 2) if len(frontiers) else None
        colors = ('#e74c3c', '#2980b9', '#27ae60', '#d99000')
        figure = Figure(figsize=(12, 6.5))
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        trajectories = np.asarray(self.trajectories)
        metadata = dict(macro_step=macro_step, coverage=float(env.explored_rate),
                        frontier_count=len(frontiers), agents=[])
        handles = []
        for axis, grid, title in zip(axes, (env.robot_belief, env.ground_truth),
                                     ('Shared Belief Map + frontiers', 'Ground Truth reference (plot only)')):
            axis.imshow(grid, cmap='gray', vmin=0, vmax=255, origin='upper')
            axis.set_title(title, fontsize=12)
            axis.set_xlim(-0.5, grid.shape[1] - 0.5)
            axis.set_ylim(grid.shape[0] - 0.5, -0.5)
            axis.set_aspect('equal')
            axis.set_axis_off()
        if frontier_cells is not None:
            axes[0].scatter(frontier_cells[:, 0], frontier_cells[:, 1], s=7, c='#ff00cc', zorder=3)
        for robot in worker.robot_list:
            agent_id = robot.id
            fov, sensor_range = env.sensor_capabilities[agent_id]
            heading = float(robot.heading)
            radius = sensor_range / info.cell_size
            color = colors[agent_id]
            trail = get_cell_position_from_coords(trajectories[:, agent_id], info).reshape(-1, 2)
            center = cells[agent_id]
            for axis in axes:
                axis.plot(trail[:, 0], trail[:, 1], color=color, linewidth=1.6, zorder=4)
                axis.add_patch(Wedge(center, radius, heading - fov / 2, heading + fov / 2,
                                     facecolor=color, edgecolor=color, alpha=0.22, zorder=5))
                direction = np.array([np.cos(np.radians(heading)), np.sin(np.radians(heading))])
                axis.annotate('', xy=center + direction * radius, xytext=center,
                              arrowprops=dict(arrowstyle='->', color=color, lw=1.4), zorder=6)
                axis.scatter(*center, s=38, color=color, edgecolors='black', linewidths=0.5, zorder=7)
                axis.annotate(f'U{agent_id}', center, xytext=(5, 5), textcoords='offset points',
                              color=color, fontsize=9, weight='bold', zorder=8)
            kind = 'A' if (fov, sensor_range) == (180, 8) else 'B' if (fov, sensor_range) == (70, 15) else '?'
            handles.append(Line2D([], [], color=color, marker='o',
                                  label=f'UAV {agent_id} | {kind} | {fov:g}\N{DEGREE SIGN} / {sensor_range:g} m'))
            metadata['agents'].append(dict(agent_id=agent_id, type=kind, fov_deg=fov, range_m=sensor_range,
                heading_deg=heading, position_m=positions[agent_id].tolist(), center_cell=center.tolist(),
                radius_cells=radius, theta1=heading - fov / 2, theta2=heading + fov / 2,
                trajectory_points=len(trail)))
        figure.suptitle(f'MARVEL V1 | Episode {self.episode} | Macro step {macro_step} | '
                       f'Coverage {100 * env.explored_rate:.2f}%\nMap: {Path(env.map_path).name}', fontsize=13)
        figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.035), ncol=4, fontsize=9)
        figure.text(0.5, 0.015, 'Magenta: Belief frontiers. Sectors: nominal FoV/range; walls can occlude scans.',
                    ha='center', fontsize=9)
        figure.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.13, wspace=0.06)
        return figure, metadata

    def finish(self, worker):
        started = time.perf_counter()
        output = make_gif(str(self.directory), f'{self.filename_prefix}episode_{self.episode:06d}_worker_{self.worker_id}',
                          self.frame_files, worker.env.explored_rate, frame_duration=0.5)
        result = dict(episode=self.episode, worker_id=self.worker_id, map_path=worker.env.map_path,
                      gif_path=output, frame_count=len(self.frame_files), frame_every=self.frame_every,
                      render_seconds=self.render_seconds, encode_seconds=time.perf_counter() - started,
                      gif_bytes=Path(output).stat().st_size, frames=self.frames)
        (self.directory / 'frames.json').write_text(json.dumps(result, indent=2))
        print('V1_GIF ' + json.dumps({key: value for key, value in result.items() if key != 'frames'}), flush=True)
        return result
