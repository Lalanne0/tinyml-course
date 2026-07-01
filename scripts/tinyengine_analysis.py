import os
import sys
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def analyze_tflite(tflite_path):
    """
    Simule une analyse statique d'un fichier .tflite avec TinyEngine.
    En l'absence de l'API complete compilee, cette fonction parse le graphe 
    pour estimer les tailles memoire de maniere simplifiee, pedagogique.
    
    Pour un projet reel, on utiliserait le CodeGenerator de TinyEngine.
    """
    if not os.path.exists(tflite_path):
        logging.error(f"Fichier non trouve: {tflite_path}")
        return None
        
    try:
        # Import local pour eviter les erreurs si le script est importe sans tflite_runtime
        import tflite_runtime.interpreter as tflite
        import numpy as np
        
        interpreter = tflite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        tensor_details = interpreter.get_tensor_details()
        
        # Estimation de la Flash (taille du modele statique)
        flash_size_bytes = os.path.getsize(tflite_path)
        
        # Estimation du pic SRAM (analyse simplifiee)
        # Dans la realite, TinyEngine utilise GeneralMemoryScheduler pour optimiser
        # le chevauchement des buffers. Ici on fait une estimation grossiere pedagogique.
        
        max_sram_bytes = 0
        current_sram_bytes = 0
        
        # On calcule la taille de chaque tensor
        tensor_sizes = {}
        for tensor in tensor_details:
            shape = tensor['shape']
            dtype = tensor['dtype']
            
            # Taille en bytes (1 pour int8/uint8, 4 pour float32)
            bytes_per_elem = 1 if dtype in [np.int8, np.uint8] else 4
            num_elems = np.prod(shape) if len(shape) > 0 else 0
            size_bytes = num_elems * bytes_per_elem
            
            tensor_sizes[tensor['index']] = size_bytes
            
        # Simulation d'un execution memory schedule tres basique
        # On prend la somme max de deux tensors consecutifs pour simuler ping-pong buffer
        sorted_tensors = sorted(tensor_details, key=lambda x: x['index'])
        
        peak_buffer = 0
        for i in range(len(sorted_tensors) - 1):
            t1 = sorted_tensors[i]
            t2 = sorted_tensors[i+1]
            
            combined_size = tensor_sizes[t1['index']] + tensor_sizes[t2['index']]
            if combined_size > peak_buffer:
                peak_buffer = combined_size
                
        # Ajout d'un overhead (variables globales, pile d'execution)
        overhead = 32 * 1024 # 32 KB overhead typique
        
        sram_size_bytes = peak_buffer + overhead
        
        return {
            'flash_bytes': flash_size_bytes,
            'flash_kb': flash_size_bytes / 1024,
            'sram_peak_bytes': sram_size_bytes,
            'sram_peak_kb': sram_size_bytes / 1024,
            'num_tensors': len(tensor_details),
            'input_shape': input_details[0]['shape'].tolist() if len(input_details) > 0 else []
        }
        
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse: {str(e)}")
        return None

def print_report(results, target_mcu="STM32F746"):
    """Affiche un rapport formate"""
    if results is None:
        return
        
    mcu_specs = {
        "STM32F746": {"sram_kb": 320, "flash_kb": 1024},
        "STM32H743": {"sram_kb": 512, "flash_kb": 2048}
    }
    
    spec = mcu_specs.get(target_mcu, {"sram_kb": 0, "flash_kb": 0})
    
    # Simuler budget OS/Firmware (reserve)
    os_reserve_sram = 50 # 50 KB reserve pour OS/stack
    os_reserve_flash = 250 # 250 KB reserve pour FW
    
    available_sram = spec['sram_kb'] - os_reserve_sram
    available_flash = spec['flash_kb'] - os_reserve_flash
    
    print("=" * 50)
    print(f"RAPPORT D'ANALYSE DE DEPLOIEMENT (Simule)")
    print("=" * 50)
    print(f"Cible Hardware    : {target_mcu}")
    print(f"SRAM Totale       : {spec['sram_kb']} KB")
    print(f"SRAM Disponible   : {available_sram} KB (apres OS)")
    print(f"Flash Totale      : {spec['flash_kb']} KB")
    print(f"Flash Disponible  : {available_flash} KB (apres OS)")
    print("-" * 50)
    print(f"Analyse du Modele :")
    print(f"Resolution Entree : {results['input_shape']}")
    print(f"Pic SRAM estime   : {results['sram_peak_kb']:.1f} KB")
    print(f"Flash requise     : {results['flash_kb']:.1f} KB")
    print("-" * 50)
    
    sram_ok = results['sram_peak_kb'] <= available_sram
    flash_ok = results['flash_kb'] <= available_flash
    
    print("CONCLUSION:")
    if sram_ok and flash_ok:
        print("✅ Le modele PEUT etre deploye sur cette cible.")
    else:
        print("❌ ECHEC du deploiement.")
        if not sram_ok:
            print(f"   -> Debordement SRAM : depassement de {results['sram_peak_kb'] - available_sram:.1f} KB")
        if not flash_ok:
            print(f"   -> Debordement Flash : depassement de {results['flash_kb'] - available_flash:.1f} KB")
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tinyengine_analysis.py <chemin_vers_modele.tflite> [CIBLE_MCU]")
        sys.exit(1)
        
    tflite_file = sys.argv[1]
    mcu = sys.argv[2] if len(sys.argv) > 2 else "STM32F746"
    
    res = analyze_tflite(tflite_file)
    print_report(res, mcu)
