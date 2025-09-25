from typing import List, Tuple, Dict, Set
from heapq import heappush, heappop
from grid_env import GridEnv
import random
import math

random.seed(42)

State = Tuple[int, int, int]  # (r, c, t)

class Planner:
    @staticmethod
    def _reconstruct_path(parent: Dict[State, State], state: State) -> List[State]:
        path = []
        current = state
        while current in parent:
            path.append(current)
            current = parent[current]
        path.append(current)  # Start
        return path[::-1]

    @staticmethod
    def bfs(env: GridEnv) -> Tuple[List[State], float, int]:
        """BFS: Shortest path in steps (cost=1 per move, ignores terrain)."""
        queue: List[State] = [(env.start[0], env.start[1], 0)]
        visited: Set[State] = set()
        parent: Dict[State, State] = {}
        nodes_expanded = 0

        while queue:
            r, c, t = queue.pop(0)
            state = (r, c, t)
            if state in visited:
                continue
            visited.add(state)
            nodes_expanded += 1

            if (r, c) == env.goal:
                path = Planner._reconstruct_path(parent, state)
                cost = len(path) - 1  # Number of steps
                return path, float(cost), nodes_expanded  # Float for consistency

            obs = env.get_obstacle_positions(t + 1)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                nstate = (nr, nc, t + 1)
                if env.is_valid(nr, nc) and nstate not in visited and (nr, nc) not in obs:
                    queue.append(nstate)
                    parent[nstate] = state
        return [], float('inf'), nodes_expanded

    @staticmethod
    def ucs(env: GridEnv) -> Tuple[List[State], float, int]:
        """UCS: Lowest cumulative terrain cost."""
        pq = [(0.0, env.start[0], env.start[1], 0)]  # (g, r, c, t)
        visited: Dict[State, float] = {}
        parent: Dict[State, State] = {}
        nodes_expanded = 0

        while pq:
            g, r, c, t = heappop(pq)
            state = (r, c, t)
            if state in visited and visited[state] <= g:
                continue
            visited[state] = g
            nodes_expanded += 1

            if (r, c) == env.goal:
                path = Planner._reconstruct_path(parent, state)
                return path, g, nodes_expanded

            obs = env.get_obstacle_positions(t + 1)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if env.is_valid(nr, nc) and (nr, nc) not in obs:
                    move_cost = env.get_cost(nr, nc)
                    ng = g + move_cost
                    nstate = (nr, nc, t + 1)
                    if nstate not in visited or ng < visited[nstate]:
                        visited[nstate] = ng
                        heappush(pq, (ng, nr, nc, t + 1))
                        parent[nstate] = state
        return [], float('inf'), nodes_expanded

    @staticmethod
    def astar(env: GridEnv) -> Tuple[List[State], float, int]:
        """A*: UCS + admissible Manhattan heuristic."""
        def h(r: int, c: int) -> float:
            return (abs(r - env.goal[0]) + abs(c - env.goal[1])) * 1.0  # Min cost = 1

        pq = [(0.0, 0.0, env.start[0], env.start[1], 0)]  # (f, g, r, c, t)
        visited: Dict[State, float] = {}
        parent: Dict[State, State] = {}
        nodes_expanded = 0

        while pq:
            _, g, r, c, t = heappop(pq)
            state = (r, c, t)
            if state in visited and visited[state] <= g:
                continue
            visited[state] = g
            nodes_expanded += 1

            if (r, c) == env.goal:
                path = Planner._reconstruct_path(parent, state)
                return path, g, nodes_expanded

            obs = env.get_obstacle_positions(t + 1)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if env.is_valid(nr, nc) and (nr, nc) not in obs:
                    move_cost = env.get_cost(nr, nc)
                    ng = g + move_cost
                    nf = ng + h(nr, nc)
                    nstate = (nr, nc, t + 1)
                    if nstate not in visited or ng < visited[nstate]:
                        visited[nstate] = ng
                        heappush(pq, (nf, ng, nr, nc, t + 1))
                        parent[nstate] = state
        return [], float('inf'), nodes_expanded

    @staticmethod
    def simulated_annealing(env: GridEnv, current_pos: Tuple[int, int] = None, max_restarts: int = 5, max_iters: int = 1000) -> Tuple[List[State], float, int]:
        """SA for local search/replanning. Perturbs path, accepts probabilistically."""
        if current_pos is None:
            current_pos = env.start
        current_t = env.time

        def generate_initial_path(start_pos: Tuple[int, int], start_t: int) -> List[State]:
            path = [(start_pos[0], start_pos[1], start_t)]
            r, c = start_pos
            t = start_t
            while (r, c) != env.goal:
                if r < env.goal[0]:
                    r += 1
                elif r > env.goal[0]:
                    r -= 1
                elif c < env.goal[1]:
                    c += 1
                else:
                    c -= 1
                t += 1
                path.append((r, c, t))
            return path

        def energy(path: List[State]) -> float:
            total = 0.0
            for i in range(1, len(path)):
                pr, pc, pt = path[i]
                if (pr, pc) in env.get_obstacle_positions(pt):
                    return float('inf')
                total += env.get_cost(pr, pc)
                if total > env.fuel_limit:
                    return float('inf')
            if path[-1][:2] != env.goal:
                return float('inf')
            return total

        def perturb(path: List[State]) -> List[State]:
            if len(path) < 2:
                return path
            idx = random.randint(1, len(path) - 1)
            r, c, t = path[idx]
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            dr, dc = random.choice(directions)
            nr, nc = r + dr, c + dc
            if env.is_valid(nr, nc):
                new_path = path[:idx] + [(nr, nc, t)] + path[idx + 1:]
                # Check collisions in new path
                for i in range(1, len(new_path)):
                    pr, pc, pt = new_path[i]
                    if (pr, pc) in env.get_obstacle_positions(pt):
                        return path
                return new_path
            return path

        initial_path = generate_initial_path(current_pos, current_t)
        initial_energy = energy(initial_path)
        best_path = initial_path
        best_energy = initial_energy
        nodes_expanded = 0  # Approx: iters * restarts

        for restart in range(max_restarts):
            path = list(initial_path)
            temp = 100.0
            cooling_rate = 0.995
            for _ in range(max_iters):
                nodes_expanded += 1
                new_path = perturb(path)
                new_energy = energy(new_path)
                if new_energy < best