import numpy as np
import matplotlib.pyplot as plt


class Obstacle:
    def __init__(self, base, sizes):
        self.base = base
        self.sizes = sizes
        self.corners = [base, base + sizes]

    def is_colliding(self, point):
        p0, p1 = self.corners
        return np.all(point > p0) and np.all(point < p1)


class World:
    # cubic world containing obstacles
    def __init__(self,
                 dim,
                 num_obstacles,
                 obstacle_area_max=0.05,
                 obstacle_size_min=0.01,
                 obstacle_ratio_max=2):
        self.dim = dim
        self.obstacle_area_max = obstacle_area_max
        self.obstacle_size_min = obstacle_size_min
        self.obstacle_max_ratio = obstacle_ratio_max
        self.obstacles = [
            self.generate_obstacle(
                dim, obstacle_area_max, obstacle_size_min, obstacle_ratio_max
            )
            for _ in range(num_obstacles)
        ]

    @staticmethod
    def generate_obstacle(dim, area_max, size_min, ratio_max):
        det = area_max
        sizes = 1
        while det >= area_max or np.min(sizes) < size_min:
            base, sizes = np.random.uniform(size=(2, dim))
            # sizes with ratio above ratio_max value are thresholded
            ratios = sizes / (np.min(sizes) + 1e-8)
            sizes[ratios >= ratio_max] = np.min(sizes) * ratio_max
            det = np.product(sizes)
        return Obstacle(base, sizes)

    def is_freespace_point(self, point):
        # check if the point is not colliding with any of the obstacles
        return not np.any(
            [obstacle.is_colliding(point) for obstacle in self.obstacles]
        )

    def is_freespace_path(self, x0, x1, delta):
        assert delta > 0
        n = int(np.ceil(np.linalg.norm(x1 - x0) / delta))
        points = np.linspace(x0, x1, n)
        return np.all([self.is_freespace_point(point) for point in points])

    def _draw_obstacle(self, ax, obstacle):
        # work only in the 2D case
        color = 'darkgray'
        base = obstacle.base
        a, b = obstacle.sizes
        corners = np.array(
            [base, base + [a, 0], base + [a, b], base + [0, b], base])
        ax.fill(corners[:, 0], corners[:, 1], color=color)

    def draw(self, ax):
        # work only in the 2D case
        for obstacle in self.obstacles:
            self._draw_obstacle(ax, obstacle)


class RRT:
    def __init__(self, world):
        self.world = world
        self.adjacence = None
        self.points = None
        self.depth = None

    @staticmethod
    def get_nearest_neighbor(x0, vertices):
        idx_neighbor = np.argmin([np.linalg.norm(v - x0) for v in vertices])
        vertex = vertices[idx_neighbor]
        return idx_neighbor, vertex

    def generate_tree(self, root, K, d_max, path_checking_delta=0.01):
        world = self.world
        vertices = np.zeros((K, root.shape[0]))
        vertices[0] = root
        counter = 1
        # adjacence matrix
        A = np.zeros((K, K), dtype=bool)
        depth = np.zeros(K)
        while counter < K:
            sample = np.random.uniform(size=root.shape[0])
            # get vertex nearest to the sample
            idx_neighbor, vertex = self.get_nearest_neighbor(
                sample, vertices[:counter]
            )
            # normalize sample
            sample = (
                vertex +
                min(d_max / np.linalg.norm(sample - vertex), 1.) *
                (sample - vertex)
            )

            if (
                world.is_freespace_point(sample) and
                world.is_freespace_path(vertex, sample, path_checking_delta)
            ):
                vertices[counter] = sample
                A[idx_neighbor, counter] = 1
                depth[counter] = depth[idx_neighbor] + 1
                counter += 1
        self.adjacence = A
        self.vertices = vertices
        self.depth = depth

    def draw_tree(self, ax):
        assert len(self.vertices) > 1
        max_depth = np.max(self.depth)
        idxs_i, idxs_j = np.where(self.adjacence)
        for i, j in zip(idxs_i, idxs_j):
            p0 = self.vertices[i]
            p1 = self.vertices[j]
            color = (self.depth[i] + self.depth[j]) / (2 * max_depth)
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]],
                color=(color, 0, 1 - color)
            )


dim = 2
world = World(dim=dim, num_obstacles=80, obstacle_area_max=0.02)
rrt = RRT(world)
root = np.random.uniform(size=dim)
while not world.is_freespace_point(root):
    root = np.random.uniform(size=dim)
rrt.generate_tree(root=root, K=10000, d_max=0.2)

fig, ax = plt.subplots(1, 1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

world.draw(ax)
rrt.draw_tree(ax)
ax.scatter(root[0], root[1], color=(0, 0.5, 0), s=100)
plt.show()
