"""Command line interface of gasim."""
import argparse


def _run(args: argparse.Namespace):
    """
    Run a simulation: TOML configs with the new engine, Python configs with the thesis engine
    :param args: Parsed arguments with config and output
    """
    from pathlib import Path

    config_path = Path(args.config)
    if config_path.suffix == '.py':
        from gasim.legacy.simulation import load_config, simulation
        simulation(load_config(config_path))
        return 0
    if config_path.suffix != '.toml':
        raise SystemExit(f"{config_path}: expected a .toml config (new engine) or a .py config (thesis engine)")

    import numpy as np

    from gasim.analysis.statistical import eyring
    from gasim.bands import OCTAVE_BANDS
    from gasim.config import load_config
    from gasim.geometry import load_room
    from gasim.io.results import save_result
    from gasim.simulate import simulate

    if args.threads is not None:
        import numba
        if not 1 <= args.threads <= numba.config.NUMBA_NUM_THREADS:
            raise SystemExit(f"--threads must lie in 1 .. {numba.config.NUMBA_NUM_THREADS}")
        numba.set_num_threads(args.threads)

    config = load_config(config_path)
    result = simulate(config)
    output = Path(args.output) if args.output else Path('output') / config_path.stem / 'result.npz'
    save_result(result, output)

    meta = result.metadata
    parameters = result.decay()
    room, _ = load_room(config)
    reference = eyring(room, meta['speed_of_sound'], np.array(meta['air_attenuation']))

    def row(label, values):
        return f"{label:<12}" + ''.join(f'{v:>8.2f}' for v in values)

    print(f"Room        {config.model}: {meta['volume']:.1f} m3, {len(result.receivers)} receivers")
    print(f"Rays        {meta['rays']} ({meta['escaped_rays']} escaped); image sources up to order "
          f"{meta['ism_order']}: {' / '.join(map(str, meta['image_sources'][0]))} paths at R1 (by order); "
          f"{meta['timings']['simulation']:.1f} s")
    print()
    print(f"{'Band (Hz)':<12}" + ''.join(f'{band:>8.0f}' for band in OCTAVE_BANDS))
    print(row('EDT (s)', np.nanmean(parameters.edt, axis=0)))
    print(row('T20 (s)', np.nanmean(parameters.t20, axis=0)))
    print(row('T30 (s)', np.nanmean(parameters.t30, axis=0)))
    print(row('Eyring (s)', reference))
    print('T30 per receiver')
    for index, values in enumerate(parameters.t30):
        print(row(f'  R{index + 1}', values))
    print(f"Saved       {output}")
    return 0


def _eyring(args: argparse.Namespace):
    """
    Print the Sabine and Eyring reverberation time of the room in a config file
    :param args: Parsed arguments with config and volume
    """
    from gasim.legacy.Acoustic.Eyring import eyring
    from gasim.legacy.Acoustic.MaterialDict import MaterialDict
    from gasim.legacy.FileIO.ReadRhino import Read3dm
    from gasim.legacy.RoomGeometry.FieldGeometry import FieldGeometry
    from gasim.legacy.simulation import load_config, resolve_path

    config = load_config(args.config)
    material_dict = MaterialDict(config['absorptionDict'], config['scatterDict'],
                                 opt_max_res=config['opt_max_res'], model_name=config['model_name'])
    rh_plane, _, _, correct_line = Read3dm(resolve_path(config['fileDir'])).convert_file()
    sound_field = FieldGeometry(rh_plane, material_dict, correct_line)

    e_rt, s_rt, alpha_avg = eyring(sound_field, args.volume)
    print(f"Eyring Equation Reverb Time is {e_rt}")
    print(f"Sabine Equation Reverb Time is {s_rt}")
    print(f"The Average Absorption Coe is {alpha_avg}")


def _check(args: argparse.Namespace):
    """
    Load the room of a TOML config, check the model and print Sabine/Eyring reverberation times
    :param args: Parsed arguments with config
    :return: Exit code, 0 if all checks pass
    """
    import numpy as np

    from gasim.air import energy_attenuation, speed_of_sound
    from gasim.analysis.statistical import eyring, sabine
    from gasim.bands import OCTAVE_BANDS
    from gasim.config import load_config
    from gasim.geometry import load_room
    from gasim.geometry.checks import check_room

    config = load_config(args.config)
    room, skipped = load_room(config)
    report = check_room(room)

    areas = room.area_by_material()
    used = ' | '.join(f"{name} {area:.1f} m2" for name, area in areas.items() if area > 0)
    unused = [name for name, area in areas.items() if area == 0]
    print(f"Model      {config.model}")
    print(f"Polygons   {room.n_polygons} ({len(room.tri_vertices)} triangles), "
          f"{int(room.poly_flipped.sum())} normals reversed to point inwards")
    print(f"Volume     {room.volume:.1f} m3, surface {room.surface_area:.1f} m2")
    print(f"Materials  {used}" + (f" | unused: {', '.join(unused)}" if unused else ""))
    if config.surfaces:
        counts = np.bincount(room.poly_surface, minlength=room.n_surfaces)
        print('Surfaces   ' + ' | '.join(f"{name} {area:.1f} m2 ({count})" for name, area, count
                                         in zip(room.surface_names, room.area_by_surface(), counts)))
    print(f"Source     {'inside' if report.source_inside else 'OUTSIDE'}; receivers inside: "
          f"{int(report.receivers_inside.sum())} of {len(report.receivers_inside)}")
    print(f"Leaks      {report.escaped_rays} of {report.traced_rays} rays escaped")
    overlaps = ', '.join(f"{room.poly_layer[i]}/{room.poly_layer[j]} ({i}, {j})" for i, j in report.overlaps)
    print(f"Overlaps   {overlaps or 'none'}")
    print(f"Skipped    {', '.join(f'{kind} on {layer}' for layer, kind in skipped) or 'none'}")

    air = config.air
    c = speed_of_sound(air.temperature)
    m = energy_attenuation(air.temperature, air.humidity, air.pressure)
    print()
    print('Band (Hz) ' + ''.join(f'{band:>8.0f}' for band in OCTAVE_BANDS))
    print('Sabine (s)' + ''.join(f'{t:>8.2f}' for t in sabine(room, c, m)))
    print('Eyring (s)' + ''.join(f'{t:>8.2f}' for t in eyring(room, c, m)))
    print(f"Air        {air.temperature} degC, {air.humidity} % RH, c = {c:.1f} m/s")
    print('OK' if report.ok else 'CHECK FAILED')
    return 0 if report.ok else 1


def _paths(args: argparse.Namespace):
    """
    Write the image-source paths of a result to a Rhino file
    :param args: Parsed arguments with result, output, receiver and max_order
    """
    from gasim.io.results import load_result
    from gasim.io.rhino import write_paths

    result = load_result(args.result)
    if result.paths is None:
        raise SystemExit(f"{args.result} holds no image-source paths")
    count = write_paths(result.paths, args.output, receiver=args.receiver - 1, max_order=args.max_order,
                        layers=result.metadata.get('layers'))
    print(f"Wrote {count} paths of receiver {args.receiver} to {args.output}")
    return 0


def _surfaces(args: argparse.Namespace):
    """
    Print the statistics per surface of a result for one band
    :param args: Parsed arguments with result and band
    """
    import numpy as np

    from gasim.io.results import load_result

    result = load_result(args.result)
    stats = result.surfaces
    if stats is None:
        raise SystemExit(f"{args.result} holds no surface statistics (result format 3 or later)")
    bands = [int(band) for band in result.bands]
    if args.band not in bands:
        raise SystemExit(f"band must be one of {bands}")
    b = bands.index(args.band)
    ism = stats.ism_reflections.sum(axis=(0, 1))  # all receivers and orders
    columns = {
        'area m2': (stats.area, '.1f'),
        'ISM refl.': (ism, 'd'),
        'ISM /m2 rel': (stats.relative_density(ism), '.2f'),
        'ray hits /m2 rel': (stats.relative_density(stats.ray_hits[b]), '.2f'),
        'incident /m2 rel': (stats.relative_density(stats.incident[b]), '.2f'),
        'absorbed %': (100 * stats.absorbed_share()[b], '.1f'),
        'received % (mean R)': (100 * stats.received_share()[:, b].mean(axis=0), '.1f'),
    }
    width = max(len(name) for name in stats.names) + 2
    print(f"{args.result}, {args.band} Hz; 'rel' = per m2 relative to the room average (1 in a diffuse field)")
    print(f"{'surface':<{width}}" + ''.join(f'{title:>{len(title) + 3}}' for title in columns))
    for s, name in enumerate(stats.names):
        cells = []
        for title, (values, fmt) in columns.items():
            value = values[s]
            text = '-' if isinstance(value, float) and np.isnan(value) else format(value, fmt)
            cells.append(text.rjust(len(title) + 3))
        print(f"{name:<{width}}" + ''.join(cells))
    direct = 100 * (stats.direct[:, b] / (stats.received[:, b].sum(axis=-1) + stats.direct[:, b])).mean()
    print(f"{'direct sound':<{width}}" + ' ' * sum(len(t) + 3 for t in list(columns)[:-1])
          + f"{direct:.1f}".rjust(len('received % (mean R)') + 3))
    return 0


def _export_paths(args: argparse.Namespace):
    """
    Write the image-source or stochastic paths of a saved receiver to a Rhino file
    :param args: Parsed arguments with kind, pkl and output
    """
    if args.kind == 'image':
        from gasim.legacy.FileIO.im_2_rhino import im2rhino as export
    else:
        from gasim.legacy.FileIO.st_2_rhino import st2rhino as export
    export(args.pkl, args.output)


def build_parser():
    """
    Build the command line parser
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(prog='gasim', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='run a simulation from a config file')
    run.add_argument('config', help='TOML config (new engine), e.g. configs/test_room.toml, '
                                    'or Python config (thesis engine), e.g. configs/legacy/test_room.py')
    run.add_argument('--output', help='result file of the new engine, default output/<config name>/result.npz')
    run.add_argument('--threads', type=int, help='CPU threads of the new engine, default all; results do not '
                                                 'depend on it')
    run.set_defaults(func=_run)

    chk = sub.add_parser('check', help='check the room model of a TOML config, print Sabine/Eyring')
    chk.add_argument('config', help='TOML config, e.g. configs/test_room.toml')
    chk.set_defaults(func=_check)

    eyr = sub.add_parser('eyring', help='Sabine/Eyring reverberation time of a legacy config room')
    eyr.add_argument('config', help='config file')
    eyr.add_argument('--volume', type=float, required=True, help='room volume in m3')
    eyr.set_defaults(func=_eyring)

    pth = sub.add_parser('paths', help='write the image-source paths of a result to a .3dm file')
    pth.add_argument('result', help='result of gasim run, e.g. output/test_room/result.npz')
    pth.add_argument('output', help='.3dm file to write')
    pth.add_argument('--receiver', type=int, default=1, help='receiver number, 1 = first (default)')
    pth.add_argument('--max-order', type=int, help='highest reflection order to write')
    pth.set_defaults(func=_paths)

    srf = sub.add_parser('surfaces', help='print reflections and energy per surface of a result')
    srf.add_argument('result', help='result of gasim run, e.g. output/test_room/result.npz')
    srf.add_argument('--band', type=int, default=1000, help='octave band in Hz, default 1000')
    srf.set_defaults(func=_surfaces)

    exp = sub.add_parser('export-paths', help='write paths saved by the thesis engine (.pkl) to a .3dm file')
    exp.add_argument('kind', choices=['image', 'stochastic'])
    exp.add_argument('pkl', help='saved receiver, e.g. output/test_room/rec_0.pkl')
    exp.add_argument('output', help='.3dm file to write')
    exp.set_defaults(func=_export_paths)

    return parser


def main(argv=None):
    """
    Command line entry point
    :param argv: Argument list, defaults to sys.argv[1:]
    :return: Exit code
    """
    args = build_parser().parse_args(argv)
    return args.func(args) or 0
