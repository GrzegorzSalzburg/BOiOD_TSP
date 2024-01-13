import matplotlib.pyplot as plt
import time
import sys


def read_data_from_file(file_name):
    """
    Odczytuje dane z pliku zawierającego informacje o grafach.

    Parametry:
    file_name (str): Nazwa pliku zawierającego dane.

    Zwraca:
    generator: Zwraca rozmiar i grafy dla kolejnych sekcji danych.
    """
    with open(file_name, 'r') as file:
        all_data = file.read().split('data:')[1:]  # Dzielenie danych na sekcje 'data:'

        for data in all_data:
            lines = data.strip().split('\n')
            size = int(lines[0].strip())  # Pobranie rozmiaru
            graph = []

            for line in lines[1:]:
                row = list(map(int, line.strip().split()))
                graph.append(row)  # Tworzenie grafu na podstawie danych

            # yield - zwrócenie wartości bez zamykania funkcji,
            yield size, graph  # Rozmiar i graf dla kolejnych sekcji danych 


def plot_all_graphs(sizes, times_bu, times_r, percent):
    # Wykres liniowy czasów wykonania
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_bu, marker='o', label='Bottom-Up DP')
    plt.plot(sizes, times_r, marker='o', label='Recursive')
    plt.xlabel('Graph Size')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Wykres słupkowy różnic czasów wykonania
    plt.figure(figsize=(10, 6))
    diff_times = [abs(bu - r) for bu, r in zip(times_bu, times_r)]
    plt.bar(sizes, diff_times, color='skyblue')
    plt.xlabel('Graph Size')
    plt.ylabel('Time Difference (seconds)')
    plt.title('Difference in Execution Times')
    plt.grid(axis='y')
    plt.show()

    # Wykres słupkowy różnic procentowych czasów wykonania
    plt.figure(figsize=(10, 6))
    plt.bar(sizes, percent, color='skyblue')
    plt.xlabel('Graph Size')
    plt.ylabel('Time Difference (percent)')
    plt.title('Percent difference in Execution Times')
    plt.grid(axis='y')
    plt.show()

    # Wykres pudełkowy czasów obu metod
    plt.figure(figsize=(8, 6))
    plt.boxplot([times_bu, times_r], labels=['Bottom-Up DP', 'Recursive'])
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Distribution')
    plt.grid(axis='y')
    plt.show()


def tsp_recursion(graph):
    """
    Rozwiązuje problem komiwojażera za pomocą rekurencyjnego algorytmu.

    Parametry:
    graph (list): Graf reprezentowany jako macierz sąsiedztwa.

    Zwraca:
    tuple: Koszt minimalnej ścieżki i trasę.
    """
    num_cities = len(graph)
    memo = [[None] * (1 << num_cities) for _ in range(num_cities)]  # Inicjalizacja tablicy memoizacji

    def tsp_recursion_main(curr, visited):
        if visited == (1 << num_cities) - 1:  # Wszystkie wierzchołki zostały odwiedzone
            return graph[curr][0], [0]  # Powrót do wierzchołka początkowego

        if memo[curr][visited] is not None: # memoizacja
            return memo[curr][visited]

        min_cost = sys.maxsize
        min_path = None
        for nxt in range(num_cities):
            # Czy wierzchołek nie jest odwiedzony i czy istnieje krawędź
            if not (visited & (1 << nxt)) and graph[curr][nxt] != 0:
                cost, path = tsp_recursion_main(nxt, visited | (1 << nxt))  # Rekurencja dla kolejnego wierzchołka
                cost += graph[curr][nxt]
                if cost < min_cost:
                    min_cost = cost
                    min_path = [curr] + path  # Zapisywanie najkrótszej ścieżki

        memo[curr][visited] = (min_cost, min_path)
        return min_cost, min_path

    return tsp_recursion_main(0, 1)  # Rozpoczęcie od wierzchołka 0


def tsp_bottom_up(graph):
    """
    Rozwiązuje problem komiwojażera za pomocą iteracyjnego algorytmu.

    Parametry:
    dist (list): Graf reprezentowany jako macierz sąsiedztwa.

    Zwraca:
    tuple: Krotka zawierająca trasę i koszt minimalnej ścieżki.
    """
    n = len(graph)
    all_visited = (1 << n) - 1  # Maska wszystkich odwiedzonych wierzchołków

    dp = [[float('inf')] * (1 << n) for _ in range(n)]  # Tablica przechowująca najkrótsze ścieżki
    path = [[-1] * (1 << n) for _ in range(n)]  # Tablica przechowująca ścieżki

    dp[0][1] = 0  # Koszt podróży do pierwszego wierzchołka

    for mask in range(1, 1 << n):
        for u in range(n):
            if mask & (1 << u):  # Czy wierzchołek jest odwiedzony
                for v in range(n):
                    # Czy wierzchołek nie jest odwiedzony i czy istnieje krawędź
                    if mask & (1 << v) == 0 and graph[u][v] != 0:
                        new_mask = mask | (1 << v)
                        new_cost = dp[u][mask] + graph[u][v]
                        if new_cost < dp[v][new_mask]:
                            dp[v][new_mask] = new_cost
                            path[v][new_mask] = u  # Zapisywanie najkrótszej ścieżki

    # Najkrótsza ścieżka powrotna do wierzchołka początkowego
    min_cost = float('inf')
    last_node = -1
    for u in range(n):
        if dp[u][all_visited] + graph[u][0] < min_cost:
            min_cost = dp[u][all_visited] + graph[u][0]
            last_node = u

    # Odtwarzanie trasy
    mask = all_visited
    tour = [0]
    while last_node != -1:
        tour.append(last_node)
        next_node = path[last_node][mask]
        mask = mask & ~(1 << last_node)
        last_node = next_node

    return tour[::-1], min_cost  # Odwracanie trasy, aby zaczynała się od 0 i kończyła na 0


def main():
    """
    Główna funkcja programu.
    """
    file_name = "tsp_data.txt"

    # Tablice na potrzeby tworzenia wykresów
    sizes = []
    times_bu = []
    times_r = []
    percent = []

    for size, graph in read_data_from_file(file_name):

        # Wywołanie metody rekurencyjnej
        start_time_r = time.time()
        tsp_recursion(graph)
        finish_time_r = time.time()
        exc_time_r = finish_time_r - start_time_r

        # Wywołanie metody iteracyjnej
        start_time_bu = time.time()
        tsp_bottom_up(graph)
        finish_time_bu = time.time()
        exc_time_bu = finish_time_bu - start_time_bu

        time_diff = exc_time_bu - exc_time_r
        if exc_time_bu + exc_time_r != 0:
            percent_diff = (time_diff / ((exc_time_bu + exc_time_r) / 2)) * 100
        else:
            percent_diff = 0

        print(f"{size} & {exc_time_bu:.3f} & {exc_time_r:.3f} & {time_diff:.3f} & {percent_diff:.1f}")

        sizes.append(size)
        times_r.append(exc_time_r)
        times_bu.append(exc_time_bu)
        percent.append(percent_diff)

    # Call the plot function
    plot_all_graphs(sizes, times_bu, times_r, percent)


if __name__ == "__main__":
    main()





