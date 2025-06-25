import numpy as np
from sympy import lambdify, symbols


def _generate_odes(adjacency_matrix, gene_symbols, params_row):
    odes = []
    for i in range(len(gene_symbols)):
        regulated_gene = gene_symbols[i]
        production_term = float(params_row.get(f'Prod_of_{regulated_gene.name}', 50.0))
        k_deg = float(params_row.get(f'Deg_of_{regulated_gene.name}', 1.0))
        degradation_term = k_deg * regulated_gene
        regulation_product = 1
        for j in range(len(gene_symbols)):
            regulation_type = adjacency_matrix[i][j]
            if regulation_type != 0:
                regulator_gene = gene_symbols[j]
                param_prefix = f'of_{regulator_gene.name}To{regulated_gene.name}'
                s = float(params_row.get(f'Trd_{param_prefix}', 16.0))
                n = float(params_row.get(f'Num_{param_prefix}', 4.0))
                if regulation_type > 0:
                    l = float(params_row.get(f'Act_{param_prefix}', 10.0))
                    K_n_pos = s**n
                    x_n_pos = regulator_gene**n
                    numerator = l + (1.0 - l) * (K_n_pos / (x_n_pos + K_n_pos))
                    regulation_product *= numerator / l
                elif regulation_type < 0:
                    l = float(params_row.get(f'Inh_{param_prefix}', 0.1))
                    K_n_neg = s**n
                    x_n_neg = regulator_gene**n
                    reg_strength = K_n_neg / (x_n_neg + K_n_neg)
                    regulation_product *= l + (1.0 - l) * reg_strength
        ode = production_term * regulation_product - degradation_term
        print(ode)
        odes.append(ode)
    return odes


def _generate_drift_function(system_odes, gene_symbols):
    symbols_as_tuple = tuple(gene_symbols)
    f_numeric = lambdify(symbols_as_tuple, system_odes, modules="numpy")
    return lambda x: np.array(f_numeric(*x), dtype=np.float64)