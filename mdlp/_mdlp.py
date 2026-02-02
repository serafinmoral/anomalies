import numpy as np
from scipy.stats import entropy
import math

def get_cut(col, ind):
    """Calcula el punto medio entre dos valores para definir el punto de corte."""
    return (col[ind - 1] + col[ind]) / 2.0

def slice_entropy(y_slice):
    """
    Calcula la entropía de un fragmento y el número de clases únicas.
    """
    if len(y_slice) == 0:
        return 0.0, 0
    
    counts = np.bincount(y_slice)
    # Filtrar ceros para evitar errores en el cálculo de entropía
    counts = counts[counts > 0]
    probabilities = counts / len(y_slice)
    
    # scipy.stats.entropy usa logaritmo natural por defecto
    return entropy(probabilities), len(counts)

def reject_split(y, start, end, k):
    """
    Determina si se debe rechazar el corte usando el criterio MDLP.
    """
    N = float(end - start)
    ent_left, k1 = slice_entropy(y[start:k])
    ent_right, k2 = slice_entropy(y[k:end])
    ent_whole, k0 = slice_entropy(y[start:end])

    # Cálculo de la ganancia de información
    part1 = (1 / N) * ((k - start) * ent_left + (end - k) * ent_right)
    gain = ent_whole - part1
    
    # Criterio MDLP (Minimum Description Length Principle)
    entropy_diff = k0 * ent_whole - k1 * ent_left - k2 * ent_right
    delta = math.log(math.pow(3, k0) - 2) - entropy_diff
    
    return gain <= (1 / N) * (math.log(N - 1) + delta)

def find_cut(y, start, end):
    """Busca el mejor punto de corte minimizando la entropía de la partición."""
    length = end - start
    prev_entropy = float('inf')
    k = -1
    
    for ind in range(start + 1, end):
        # Optimización: Solo evaluar cortes donde cambia la clase
        if y[ind - 1] == y[ind]:
            continue

        ent_left, _ = slice_entropy(y[start:ind])
        ent_right, _ = slice_entropy(y[ind:end])
        
        curr_entropy = ((ind - start) / length * ent_left + 
                        (end - ind) / length * ent_right)

        if prev_entropy > curr_entropy:
            prev_entropy = curr_entropy
            k = ind
            
    return k

def MDLPDiscretize(col, y, min_depth, min_split):
    """
    Realiza la discretización MDLP sobre una columna X y etiquetas y.
    """
    # Ordenar datos según los valores de la columna
    order = np.argsort(col)
    col = col[order]
    y = y[order].astype(int)

    cut_points = set()
    num_samples = len(col)
    
    # Intervalos a explorar: (start, end, depth)
    search_intervals = [(0, num_samples, 0)]

    while search_intervals:
        start, end, depth = search_intervals.pop()
        
        # Si el segmento es más pequeño que el mínimo permitido, no dividir
        if end - start <= min_split:
            continue
            
        k = find_cut(y, start, end)

        # Si no hay corte posible o el criterio MDLP dice que paremos
        if k == -1 or (depth >= min_depth and reject_split(y, start, end, k)):
            front = float('-inf') if start == 0 else get_cut(col, start)
            back = float('inf') if end == num_samples else get_cut(col, end)

            if front == back: 
                continue
            if front != float('-inf'):
                cut_points.add(front)
            if back != float('inf'):
                cut_points.add(back)
            continue

        # Agregar los nuevos intervalos a la pila (DFS)
        search_intervals.append((k, end, depth + 1))
        search_intervals.append((start, k, depth + 1))

    return np.sort(np.array(list(cut_points)))