"""Standalone validation/test, exact confusion matrix and optional georeferenced predictions."""
import argparse
from pathlib import Path
import torch
from kmap.common import load_config, dump_json
from kmap.data import BuildingDataset
from kmap.model import KGVLMap
from kmap.hidden_diagnostic import HiddenTrainingDiagnostic
from train import make_loader, validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/seven_class.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--output', required=True)
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--max-batches', type=int, help='Diagnostic subset only; never report this as full validation.')
    p.add_argument('--hidden-training-diagnostic', help='Path to hidden_truth_diagnostic.json; post-training diagnosis on hidden training pixels, not independent validation/test.')
    args = p.parse_args()
    cfg = load_config(args.config)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved['config']['classes'] != cfg['classes']:
        raise ValueError('Checkpoint class definition differs from evaluation configuration.')
    model = KGVLMap(cfg, text_embeddings=saved['model']['psa.text_embeddings'], pretrained=False)
    model.load_state_dict(saved['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = HiddenTrainingDiagnostic(cfg,args.hidden_training_diagnostic) if args.hidden_training_diagnostic else BuildingDataset(cfg, args.split)
    result = validate(model, make_loader(data, cfg), device, cfg['amp'], limit=args.max_batches,
        output_dir=Path(args.output).parent / 'predictions' if args.save_predictions else None, classes=cfg['classes'])
    result['checkpoint'] = str(Path(args.checkpoint).resolve())
    result['split'] = 'hidden_training_diagnostic' if args.hidden_training_diagnostic else args.split
    if args.hidden_training_diagnostic:
        result['evaluation_scope'] = 'Originally hidden training-building pixels only; background true pixels are absent from this diagnostic mask.'
        result['independent_test_set'] = False
        result['use_for_checkpoint_or_hyperparameter_selection'] = False
    result['partial_evaluation'] = args.max_batches is not None
    dump_json(args.output, result)
    print({k: result[k] for k in ['OA', 'Macro-P', 'Macro-R', 'Macro-F1', 'Kappa']})


if __name__ == '__main__':
    main()
