#!/usr/bin/env python3
"""
newton_graph_9_4.py

Прототип реализации §9.4: решение разностной схемы методом Ньютона
на графе сосудов. Для простоты:
 - плотный Якобиан (для тестовых небольших графов)
 - каждое ребро — отдельный одномерный сегмент с одинаковыми физ.параметрами
 - в вершине реализовано простое сопряжение: одинаковое давление + сумма расходов = 0
 - граничные условия: заданное p_in на одном внешнем конце, свободный выход на остальных

Как использовать:
 - Запустить, взять результаты (сохранится .npz и распечатаются логи).
 - Для ускорения/масштабирования переписать якобиан в sparse.
"""

import numpy as np
import time
import math
import os

# -------------------------
# Физика / число
# -------------------------
rho = 1060.0
nu = 0.0005
g = 9.81
phi = 0.0

S0 = 1.0e-4
p0 = 13300.0
K = 2e5

alpha = 0.4
# общий временной шаг
tau = 5e-5
sigma = 0.6

# Newton parameters
newton_tol = 1e-6
newton_maxiter = 20
jac_eps = 1e-7

# -------------------------
# Структура графа (пример)
# -------------------------
# Удобно хранить ребро как dict: { 'L':..., 'Nx':..., 'h':..., 'id':int }
# Пример: тройник: ребро 0 (центральный L=1), к его правому концу примыкают ребра 1 и 2
edges = []
# центральное ребро (идёт от входа к узлу)
edges.append({'id':0, 'L':1.0, 'Nx':21})
# два ответвления (от узла к выходам)
edges.append({'id':1, 'L':0.5, 'Nx':11})
edges.append({'id':2, 'L':0.5, 'Nx':11})

# Вершины: каждая вершина — список кортежов (edge_id, local_index)
# локальный индекс: 0 — левый конец ребра, Nx-1 — правый конец ребра
# Здесь: вершина 0 = вход (левый конец edge0)
#       вершина 1 = соединение: right end of edge0 (index Nx0-1), left end of edge1 (0), left of edge2 (0)
#       вершины 2 and 3 = свободные выходы (right ends of edges 1 and 2)
vertices = []
vertices.append([ (0,0) ])  # вход
vertices.append([ (0, edges[0]['Nx']-1), (1,0), (2,0) ])  # тройник: соединение
vertices.append([ (1, edges[1]['Nx']-1) ])  # выход 1
vertices.append([ (2, edges[2]['Nx']-1) ])  # выход 2

# Маркируем, какие вершины граничные и с какими условиями:
# vertex_bc: dict: vertex_id -> {'type':'dirichlet'/'neumann'/'junction', ...}
vertex_bc = {
    0: {'type':'dirichlet', 'p_in_func': lambda t: p0 + 800.0*math.sin(30.0*math.pi*t)},  # вход: заданное давление
    1: {'type':'junction'},  # внутреннее соединение — условия сопряжения
    2: {'type':'free'},      # свободный выход (открытый конец)
    3: {'type':'free'}
}

# -------------------------
# Подготовка ребер (пространственная сетка)
# -------------------------
for e in edges:
    e['h'] = e['L'] / (e['Nx'] - 1)
    e['x'] = np.linspace(0.0, e['L'], e['Nx'])

# -------------------------
# Вспомогательные (из одно-ребра реализации)
# -------------------------
def S_of_p(p_arr):
    return S0 * (1.0 + (p_arr - p0) / K)

def laplacian(arr, h):
    lap = np.zeros_like(arr)
    lap[1:-1] = (arr[2:] - 2.0*arr[1:-1] + arr[:-2]) / (h*h)
    lap[0] = (arr[2] - 2.0*arr[1] + arr[0]) / (h*h)
    lap[-1] = (arr[-1] - 2.0*arr[-2] + arr[-3]) / (h*h)
    return lap

def compute_F_fields_edge(S_arr, u_arr, p_arr, h_local, a_s, a_u):
    # spatial derivatives
    dSdx = np.zeros_like(S_arr)
    dudx = np.zeros_like(u_arr)
    dpdx = np.zeros_like(p_arr)
    dSdx[1:-1] = (S_arr[2:] - S_arr[:-2]) / (2.0*h_local)
    dudx[1:-1] = (u_arr[2:] - u_arr[:-2]) / (2.0*h_local)
    dpdx[1:-1] = (p_arr[2:] - p_arr[:-2]) / (2.0*h_local)
    dSdx[0] = (S_arr[1] - S_arr[0]) / h_local
    dudx[0] = (u_arr[1] - u_arr[0]) / h_local
    dpdx[0] = (p_arr[1] - p_arr[0]) / h_local
    dSdx[-1] = (S_arr[-1] - S_arr[-2]) / h_local
    dudx[-1] = (u_arr[-1] - u_arr[-2]) / h_local
    dpdx[-1] = (p_arr[-1] - p_arr[-2]) / h_local

    lap_s = laplacian(S_arr, h_local)
    lap_u = laplacian(u_arr, h_local)
    S_safe = np.maximum(S_arr, 1e-8)

    F_S = -(u_arr * dSdx + S_arr * dudx) + a_s * lap_s
    F_u = -(u_arr * dudx + (1.0/rho) * dpdx + (8.0 * np.pi * nu * u_arr)/S_safe - g * math.cos(phi)) + a_u * lap_u
    return F_S, F_u

# -------------------------
# Индексация неизвестных в глобальном векторе
# -------------------------
# Для простоты: будем считать неизвестными ВСЕ S и u во всех узлах, включая границы.
# Но для вершин с сопряжением будут дополнительные уравнения; чтобы избежать путаницы,
# проще включить все S_j, u_j в вектор y, а в G задать дополнительные уравнения в вершинах.
# Индексы: для каждого ребра e, для локального узла j: we map to global index base[e] + j
edge_var_ptr = {}  # e.id -> base_index in global vector for S and for u
total_nodes = 0
for e in edges:
    n = e['Nx']
    edge_var_ptr[e['id']] = {'base_S': total_nodes, 'N': n}
    total_nodes += n
# second block for u
base_u = total_nodes
for e in edges:
    edge_var_ptr[e['id']]['base_u'] = base_u
    base_u += e['Nx']
total_unknowns = base_u  # total S + total u

def pack_state(edges_fields):
    """edges_fields: list of (S,u,p) for each edge in order edges[]. returns global vector y"""
    y = np.zeros(total_unknowns)
    for e_idx, e in enumerate(edges):
        eid = e['id']
        S, u, p = edges_fields[e_idx]
        baseS = edge_var_ptr[eid]['base_S']
        y[baseS:baseS+e['Nx']] = S
    for e_idx, e in enumerate(edges):
        eid = e['id']
        S, u, p = edges_fields[e_idx]
        baseU = edge_var_ptr[eid]['base_u']
        y[baseU:baseU+e['Nx']] = u
    return y

def unpack_state(y):
    """return list of (S,u,p) per edge"""
    res = []
    for e in edges:
        eid = e['id']
        n = e['Nx']
        baseS = edge_var_ptr[eid]['base_S']
        baseU = edge_var_ptr[eid]['base_u']
        S = y[baseS:baseS+n].copy()
        u = y[baseU:baseU+n].copy()
        p = p0 + K*(S/S0 - 1.0)
        res.append((S,u,p))
    return res

# -------------------------
# Построение невязки G(y) для всего графа (включая уравнения сопряжения)
# -------------------------
def build_global_residual(y_old, y_new, t):
    """
    y_old, y_new: глобальные векторы (S and u for all edges)
    t: текущее время (для задания p_in)
    Возвращает вектор G длины total_unknowns + N_vertex_eqs (но мы будем использовать что G длины total_unknowns,
    реализуя сопряжение через уравнения, наложенные на соответствующие компоненты)
    """
    # для простоты: мы формируем G такого же размера, что и y (S and u), и в вершинах перезаписываем
    # соответствующие компоненты в G тематическими уравнениями (равенство давлений, суммарный расход = 0 и т.д.)
    G = np.zeros(total_unknowns)
    # параметры искусственной вязкости локально (можно варьировать по ребру)
    for e_idx,e in enumerate(edges):
        n = e['Nx']
        h_local = e['h']
        baseS = edge_var_ptr[e['id']]['base_S']
        baseU = edge_var_ptr[e['id']]['base_u']
        S_old = y_old[baseS:baseS+n]
        u_old = y_old[baseU:baseU+n]
        p_old = p0 + K*(S_old/S0 - 1.0)
        S_new = y_new[baseS:baseS+n]
        u_new = y_new[baseU:baseU+n]
        p_new = p0 + K*(S_new/S0 - 1.0)

        a_u = alpha * h_local * math.sqrt(K/rho)
        a_s = a_u * 0.1

        F_S_old, F_u_old = compute_F_fields_edge(S_old, u_old, p_old, h_local, a_s, a_u)
        F_S_new, F_u_new = compute_F_fields_edge(S_new, u_new, p_new, h_local, a_s, a_u)

        # невязки по внутренним узлам (включая границы)
        for j in range(n):
            GS = S_new[j] - S_old[j] - tau*((1.0 - sigma)*F_S_old[j] + sigma*F_S_new[j])
            GU = u_new[j] - u_old[j] - tau*((1.0 - sigma)*F_u_old[j] + sigma*F_u_new[j])
            G[baseS + j] = GS
            G[baseU + j] = GU

    # Теперь заменим/добавим уравнения сопряжения в вершинах.
    # Набор условий, которые часто используют:
    # 1) давления на ко-входах вершины равны друг другу (p_a = p_b = p_vertex)
    # 2) суммарный расход (sum S*u with appropriate sign) в вершине = 0 (сохранение массы)
    # Мы реализуем их, перезаписывая соответствующие строки G для первого упоминания p, и добавляя остаточные уравнения для остальных.
    # Для этого нам нужно выбрать, какие компоненты y соответствуют p — у нас p вычисляется из S, поэтому равенство p -> равенство S.

    # Проходим по вершинам:
    for v_idx, conn in enumerate(vertices):
        bc = vertex_bc.get(v_idx, {'type':'junction'})
        if bc['type'] == 'dirichlet':
            # задать p в этой вершине: конвертируем p->S и наложим GC: S_local - S_prescribed = 0
            # предположим единственный связанный (edge,loc)
            (e_id, loc) = conn[0]
            # заданное p:
            p_val = bc['p_in_func'](t)
            S_pres = S0 * (1.0 + (p_val - p0)/K)
            baseS = edge_var_ptr[e_id]['base_S']
            # записываем уравнение в соответствующую строку G[baseS + loc] = S_new - S_pres
            G[baseS + loc] = y_new[baseS + loc] - S_pres
            # и для скорости (можно держать u free или задать характер.форма; оставим уравнение из схемы для u)
        elif bc['type'] == 'free':
            # свободный выход: можно оставить как есть (уравнения по ребру уже содержат условие p[-1]=p[-2] из схемы),
            # но часто задают экстра условие p_end=p_end_prev; оставим как есть.
            pass
        else:
            # junction: implement equal pressures + sum(flows)=0
            # choose reference: first connection defines reference pressure
            ref_e, ref_loc = conn[0]
            baseS_ref = edge_var_ptr[ref_e]['base_S']
            Sref_idx = baseS_ref + ref_loc
            # 1) для всех остальных соединений накладываем равенство давления: S(e_i,loc_i) - Sref = 0
            for (e_id, loc) in conn[1:]:
                baseS = edge_var_ptr[e_id]['base_S']
                idx = baseS + loc
                # We overwrite residual at idx to be equality cond
                G[idx] = y_new[idx] - y_new[Sref_idx]
            # 2) суммарный расход: sum(sign * S*u) = 0
            # определяем направление: если локальный индекс == 0 (левый конец) — поток направлен наружу для ребра,
            # если локальный == Nx-1 (правый конец) — поток направлен внутрь. Конвенция: положительный поток в сторону возрастания x.
            # Для вершины с примыканием концов мы ставим знаки так, чтобы outgoing positive -> sum = 0
            flow_sum = 0.0
            # We'll write this equation into residual of first conn's u-component for uniqueness
            # Calculate sum:
            for (e_id, loc) in conn:
                baseS = edge_var_ptr[e_id]['base_S']
                baseU = edge_var_ptr[e_id]['base_u']
                Sval = y_new[baseS + loc]
                Uval = y_new[baseU + loc]
                # We need sign: if loc==0 then flow into vertex is ( - S*U )? We'll adopt:
                # flow_out_of_vertex = + S * u if u goes outwards. To be consistent, we take:
                # if loc == 0 (left end), flow toward right is positive; flow entering vertex from this edge is -S*u
                # if loc == Nx-1 (right end), flow entering vertex from this edge is +S*u
                # So sum_in = sum( sign_in * S*U ), and we want sum_in = 0
                Nloc = edges[e_id]['Nx']
                if loc == 0:
                    sign = -1.0
                elif loc == Nloc - 1:
                    sign = 1.0
                else:
                    # interior connection? not expected; treat as zero
                    sign = 0.0
                flow_sum += sign * Sval * Uval
            # Write flow residual into G at the u-component of the reference (first) connection
            ref_eid, ref_loc = conn[0]
            baseUref = edge_var_ptr[ref_eid]['base_u']
            u_idx = baseUref + ref_loc
            G[u_idx] = flow_sum

    return G

# -------------------------
# Newton outer driver (one time step)
# -------------------------
def newton_step(y_old, t):
    # initial guess: explicit predictor
    # do a single explicit predictor for each edge
    y_guess = y_old.copy()
    # explicit predictor per edge:
    for e in edges:
        eid = e['id']
        n = e['Nx']
        baseS = edge_var_ptr[eid]['base_S']
        baseU = edge_var_ptr[eid]['base_u']
        S_old = y_old[baseS:baseS+n]
        u_old = y_old[baseU:baseU+n]
        p_old = p0 + K*(S_old/S0 - 1.0)
        h_local = e['h']
        a_u = alpha * h_local * math.sqrt(K/rho)
        a_s = a_u * 0.1
        F_S_old, F_u_old = compute_F_fields_edge(S_old, u_old, p_old, h_local, a_s, a_u)
        S_pred = np.maximum(S_old + tau * F_S_old, 1e-8)
        u_pred = u_old + tau * F_u_old
        y_guess[baseS:baseS+n] = S_pred
        y_guess[baseU:baseU+n] = u_pred

    # Newton iterations
    for it in range(newton_maxiter):
        G = build_global_residual(y_old, y_guess, t)
        normG = np.linalg.norm(G)
        if normG < newton_tol:
            return y_guess, True, it, normG
        # numeric jacobian
        N = y_guess.size
        J = np.zeros((N,N))
        # For performance use columns on small problems; for bigger use sparse
        for col in range(N):
            y_try = y_guess.copy()
            y_try[col] += jac_eps
            G_try = build_global_residual(y_old, y_try, t)
            J[:,col] = (G_try - G) / jac_eps
        # solve
        try:
            delta = np.linalg.solve(J, -G)
        except np.linalg.LinAlgError:
            return y_guess, False, it, normG
        y_guess += delta
        # protections
        # clip S positive
        # clip u reasonable (optional)
        # y_guess[:total_S] = np.clip(y_guess[:total_S], 1e-8, None)
    return y_guess, False, newton_maxiter, np.linalg.norm(build_global_residual(y_old, y_guess, t))

# -------------------------
# Инициализация и основной цикл по времени
# -------------------------
def run_graph_simulation(Tfinal=0.2):
    t_steps = int(Tfinal / tau) + 1
    # initial conditions: resting
    edges_fields = []
    for e in edges:
        n = e['Nx']
        p_init = np.ones(n) * p0
        S_init = S_of_p(p_init)
        u_init = np.zeros(n)
        edges_fields.append((S_init, u_init, p_init))
    y = pack_state(edges_fields)

    history = {'t':[], 'pmid_edges': [ [] for _ in edges ], 'p0': []}
    t = 0.0
    start = time.time()
    for nstep in range(t_steps):
        # time
        t = nstep * tau
        # apply dirichlet p_in by modifying y_old for the predictor and Newton (we enforce inside build_global_residual)
        y_old = y.copy()
        # Newton solve for y_new at time t+tau
        y_new, ok, iters, normG = newton_step(y_old, t+tau)
        if not ok:
            print("Newton failed at t=", t+tau, "iters", iters, "normG", normG)
            break
        y = y_new
        # unpack and logging
        edges_state = unpack_state(y)
        history['t'].append(t+tau)
        # record mid pressure for each edge
        for ei,(S,u,p) in enumerate(edges_state):
            history['pmid_edges'][ei].append(p[len(p)//2])
        # record global input pressure (vertex 0)
        p_in_val = vertex_bc[0]['p_in_func'](t+tau)
        history['p0'].append(p_in_val)

        if nstep % 200 == 0:
            print(f"[t={t+tau:.6f}] step {nstep}/{t_steps}, Newton iters={iters}, normG={normG:.2e}")

    elapsed = time.time() - start
    print("Finished, elapsed", elapsed, "s")
    return y, history

if __name__ == "__main__":
    y_final, history = run_graph_simulation(Tfinal=0.1)
    # save
    np.savez("results_graph_newton.npz", t=np.array(history['t']),
             p0=np.array(history['p0']),
             pmid0=np.array(history['pmid_edges'][0]),
             pmid1=np.array(history['pmid_edges'][1]),
             pmid2=np.array(history['pmid_edges'][2]))
    print("Saved results_graph_newton.npz")
