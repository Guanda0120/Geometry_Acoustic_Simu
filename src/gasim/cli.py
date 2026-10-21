"""Command line interface of gasim."""
import argparse


def _run(args: argparse.Namespace):
    """
    Run a simulation from a config file
    :param args: Parsed arguments with config and engine
    """
    from gasim.legacy.simulation import load_config, simulation

    simulation(load_config(args.config))


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
    run.add_argument('config', help='config file, e.g. configs/legacy/test_room.py')
    run.add_argument('--engine', choices=['legacy'], default='legacy',
                     help='propagation engine (only the thesis-2023 engine for now)')
    run.set_defaults(func=_run)

    eyr = sub.add_parser('eyring', help='Sabine/Eyring reverberation time of a config room')
    eyr.add_argument('config', help='config file')
    eyr.add_argument('--volume', type=float, required=True, help='room volume in m3')
    eyr.set_defaults(func=_eyring)

    exp = sub.add_parser('export-paths', help='write saved ray paths to a .3dm file')
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
    args.func(args)
    return 0
