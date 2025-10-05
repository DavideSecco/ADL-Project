import torch
import argparse
import os
import re
import sys


# da terminale:

# python Weights_Converter.py 
# --input_cktp_path                 # mettere il path del checkpoint DeCUR o DenseDeCUR   NO PER DenseCL!
# --input_cktp_path_visible         # mettere il path del checkpoint DenseCL VIS
# --input_cktp_path_infrared        # mettere il path del checkpoint DenseCL IR
# --model_input                     # mettere 'decur' o 'densecl' o 'decurdense'
# --output_name icafusion_from      # mettere il nome del file di output (es: icafusion_from_decur.pth)
# --output_dir                      # mettere la directory di output (default ./ICAFusion/final_checkpoints/)
# --icafusion_base                  # mettere il path del checkpoint base di ICAFusion (default /mnt/proj3/eu-25-19/davide_secco/ADL-Project/ICAFusion/state_dict_from_model.pth)



# Funzione di mapping da DeCUR a ICAFusion

def map_decur_to_ica_semantic(icafusion_state_dict, decur_state_dict, backbone_id=1):
  
    # Il risultato finale (new_state_dict) conterrà tutto: pesi backbone aggiornati e tutto il resto come prima.
    new_state_dict = icafusion_state_dict.copy()
    # Su quale backbone di DeCUR lavoriamo (1 o 2) 
    decur_prefix = f"module.backbone_{backbone_id}."

    # Offset backbone 1 = model.0-4, backbone 2 = model.5-9 come mostrato sopra
    ica_offset = 0 if backbone_id == 1 else 5
    matched, skipped = [], []

    for k_decur, v_decur in decur_state_dict.items():
        if not k_decur.startswith(decur_prefix):  # skippa chiavi non della backbone corretta
            continue
        # esempio: da module.backbone_1.conv1.weight  a conv1.weight 
        # da module.backbone_1.layer1.0.conv1.weight a layer1.0.conv1.weight
        short = k_decur.replace(decur_prefix, '')

        # Conv1/Bn1 (iniziali)   i primi 5 layer in DECUR
        if short.startswith('conv1'):
            k_ica = f"model.{ica_offset}.layer.0" + short[len('conv1'):]
        elif short.startswith('bn1'):
            k_ica = f"model.{ica_offset}.layer.1" + short[len('bn1'):]
        # Blocchi residui layer1-4 (es: layer1.0.conv1.weight)  i restanti layer in DECUR
        elif short.startswith('layer'):
            # Estraggo blocco/resid interno:
            # esempio: layer1.0.conv2.weight
            m = re.match(r"layer(\d)\.(\d+)\.(.*)", short)
            if not m:
                skipped.append((k_decur, "no regex match"))
                continue
            block_num, block_idx, sublayer = m.groups()
            k_ica = f"model.{ica_offset + int(block_num)}.layer.{block_idx}.{sublayer}"
            
            # Correzione unica differenza di nomenclatura layer. 
            # DeCUR usa 'downsample' mentre ICAfusion usa 'shortcut'
            k_ica = k_ica.replace('downsample', 'shortcut')
        else:
            skipped.append((k_decur, "no mapping rule"))
            continue

        # Check presenza e shape!
        if k_ica in icafusion_state_dict and icafusion_state_dict[k_ica].shape == v_decur.shape:
            new_state_dict[k_ica] = v_decur
            matched.append((k_decur, k_ica))
        else:
            skipped.append((k_decur, k_ica))

    print(f"[Backbone {backbone_id}] Matchati: {len(matched)}. Saltati: {len(skipped)}")
    if len(matched) < 10:
        print("Esempi matching:", matched[:10])
    if len(skipped) > 0:
        print("Esempi saltati:", skipped[:10])
    return new_state_dict

#############################################################################################################################

# Funzione di mapping da DenseCL a ICAFusion

def map_densecl_to_ica(icafusion_state_dict, densecl_state_dict, domain):
   
    new_state_dict = icafusion_state_dict.copy()
    prefix = "module.encoder_q.0."
    ica_offset = 0 if domain == 'visible' else 5
    matched, skipped = [], []

    for k_densecl, v_densecl in densecl_state_dict.items():
        if not k_densecl.startswith(prefix):
            continue
        short = k_densecl.replace(prefix, '')  # conv1.weight / bn1.weight / layer1.0.conv1.weight

        # Conv1/Bn1 (iniziali)
        if short.startswith('conv1'):
            k_ica = f"model.{ica_offset}.layer.0" + short[len('conv1'):]
        elif short.startswith('bn1'):
            k_ica = f"model.{ica_offset}.layer.1" + short[len('bn1'):]
        # Blocchi layer1-4
        elif short.startswith('layer'):
            m = re.match(r"layer(\d)\.(\d+)\.(.*)", short)
            if not m:
                skipped.append((k_densecl, "no regex match"))
                continue
            block_num, block_idx, sublayer = m.groups()
            k_ica = f"model.{ica_offset + int(block_num)}.layer.{block_idx}.{sublayer}"
            k_ica = k_ica.replace('downsample', 'shortcut')
        else:
            skipped.append((k_densecl, "no mapping rule"))
            continue

        if k_ica in icafusion_state_dict and icafusion_state_dict[k_ica].shape == v_densecl.shape:
            new_state_dict[k_ica] = v_densecl
            matched.append((k_densecl, k_ica))
        else:
            skipped.append((k_densecl, k_ica))

    print(f"[DenseCL dom: {domain}] Matchati: {len(matched)}. Saltati: {len(skipped)}")
    if len(matched) < 10:
        print("Esempi matching:", matched)
    if skipped:
        print("Esempi saltati:", skipped[:10])
    return new_state_dict

#############################################################################################################################

# Funzione di mapping da DenseDeCUR a ICAFusion

def map_dense_decur_to_ica(icafusion_state_dict, decdecur_state_dict, mod=1):
    """
    Popola la backbone ICAFusion dal DenseDeCUR.
    - mod=1: usa encoder_q di mod1 → backbone_1 (model.0~4)
    - mod=2: usa encoder_q di mod2 → backbone_2 (model.5~9)
    """
    prefix = f"module.mod{mod}.encoder_q.0."     # prefisso per mod1 o mod2 per capire quale target backbone ICA usare
    ica_offset = 0 if mod == 1 else 5
    new_state_dict = icafusion_state_dict.copy()
    matched, skipped = [], []

    for k_src, v_src in decdecur_state_dict.items():
        if not k_src.startswith(prefix):
            continue
        short = k_src.replace(prefix, '')

        # Mappature semantiche per ResNet backbone (come fatto per DenseCL)
        if short.startswith('conv1'):
            k_tgt = f"model.{ica_offset}.layer.0{short[len('conv1'):]}"
        elif short.startswith('bn1'):
            k_tgt = f"model.{ica_offset}.layer.1{short[len('bn1'):]}"
        elif short.startswith('layer'):
            m = re.match(r"layer(\d)\.(\d+)\.(.*)", short)
            if not m:
                skipped.append((k_src, 'no regex match'))
                continue
            block, idx, rest = m.groups()
            k_tgt = f"model.{ica_offset + int(block)}.layer.{idx}.{rest}"
            k_tgt = k_tgt.replace("downsample", "shortcut")
        else:
            skipped.append((k_src, "no mapping rule"))
            continue

        # Match shape finale 
        if k_tgt in icafusion_state_dict and v_src.shape == icafusion_state_dict[k_tgt].shape:
            new_state_dict[k_tgt] = v_src
            matched.append((k_src, k_tgt))
        else:
            skipped.append((k_src, k_tgt))

    print(f"[DenseDeCUR Mod {mod}] Matchati: {len(matched)}, Saltati: {len(skipped)}")
    

    return new_state_dict



#############################################################################################################################

# Funzione di salvataggio del checkpoint ICAFusion

def save_icafusion_state_dict_wrapped(icafusion_state_dict, save_path):
    to_save = {'model': icafusion_state_dict}
    torch.save(to_save, save_path)
    print(f"Checkpoint ICAfusion salvato come: {save_path}")


#############################################################################################################################


def main():
    parser = argparse.ArgumentParser(description="Converti e copia checkpoint per ICAFusion")
    parser.add_argument('--input_cktp_path', required=True, type=str, help='Path checkpoint di input relativo a locazione di questo script, SOLO PER DeCUR o DenseDeCUR')
    
    parser.add_argument('--input_cktp_path_visible', required=False, type=str, help='Path checkpoint di input relativo a locazione di questo script DenseCL VIS')
    parser.add_argument('--input_cktp_path_infrared', required=False, type=str, help='Path checkpoint di input relativo a locazione di questo script DenseCL IR')

    parser.add_argument('--model_input', required=True, type=str, choices=['decur', 'densecl', 'decurdense'], help='Tipo modello di input')
    parser.add_argument('--output_name', required=True, type=str, help='Nome file checkpoint convertito')
    parser.add_argument('--output_dir', type=str, default="./ICAFusion/final_checkpoints/", help='Directory output checkpoint')
    parser.add_argument('--icafusion_base', type=str, default="/mnt/proj3/eu-25-19/davide_secco/ADL-Project/ICAFusion/state_dict_from_model.pth", help='Checkpoint base ICAFusion')

    args = parser.parse_args()


    # Controlli per outputh e input
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)   #creo path completo output
    ## Controllo che all'output dir non esista già il file output_name
    if os.path.exists(output_path): 
        risposta = input(f"Il file {output_path} esiste già. Vuoi sovrascriverlo? [yes/no]: ").strip().lower()
        if risposta not in ["yes", "y"]:
            print("Operazione annullata. Uscita...")
            sys.exit(0)
        else:
                print("Procedo con la sovrascrittura...")

    ## Caricamento checkpoint ICAFusion base
    icafusion_model_state_dict = torch.load(args.icafusion_base, map_location='cpu')   ## carico checkpoint base ICAFusion

    if args.model_input == 'densecl':
        if not args.input_cktp_path_visible or not args.input_cktp_path_infrared:
            raise ValueError("Per DenseCL servono sia --input_cktp_path_visible che --input_cktp_path_infrared!")
        # Carica entrambi gli state dict
        data_vis = torch.load(args.input_cktp_path_visible, map_location='cpu')
        state_dict_vis = data_vis['model'] if 'model' in data_vis else data_vis
        data_ir = torch.load(args.input_cktp_path_infrared, map_location='cpu')
        state_dict_ir = data_ir['model'] if 'model' in data_ir else data_ir
    else:
        if not args.input_cktp_path:
            raise ValueError("Serve --input_cktp_path se NON usi DenseCL!")
        data = torch.load(args.input_cktp_path, map_location='cpu')
        state_dict = data['model'] if 'model' in data else data



    # Conversione
    if args.model_input == 'decur':
        print("Inizio conversione da DeCUR a ICAFusion...")
        out_state_dict = map_decur_to_ica_semantic(icafusion_model_state_dict, state_dict, backbone_id=1)
        out_state_dict = map_decur_to_ica_semantic(out_state_dict, state_dict, backbone_id=2)
    elif args.model_input == 'densecl':
        print("Inizio conversione da DenseCL a ICAFusion...")
        out_state_dict = map_densecl_to_ica(icafusion_model_state_dict, state_dict_vis, domain='visible')
        out_state_dict = map_densecl_to_ica(out_state_dict, state_dict_ir, domain='infrared')
    elif args.model_input == 'decurdense':
        print("Inizio conversione da DenseDeCUR a ICAFusion...")
        out_state_dict = map_dense_decur_to_ica(icafusion_model_state_dict, state_dict, mod=1)
        out_state_dict = map_dense_decur_to_ica(out_state_dict, state_dict, mod=2)
    else:
        raise ValueError("Tipo modello non gestito.")


    # Salvataggio
    print("\nSalvataggio checkpoint convertito...\n")
    save_icafusion_state_dict_wrapped(out_state_dict, output_path)

if __name__ == '__main__':
    main()
