import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class AirfoilCoordinates:
    """
    Class to handle airfoil coordinates and related operations.
    """
    def __init__(self, x, y_upper, y_lower):
        self.x = x
        self.y_upper = y_upper
        self.y_lower = y_lower

    def normalize(self):
        # Normalizing the airfoil coordinates to a standard form
        x_norm = self.x / max(self.x)
        y_upper_norm = self.y_upper / max(self.x)
        y_lower_norm = self.y_lower / max(self.x)
        return AirfoilCoordinates(x_norm, y_upper_norm, y_lower_norm)

    def apply_transformation(self, transformation):
        # Apply a given transformation to the coordinates
        return transformation.apply(self)

class Airfoil:
    def __init__(self, designation):
        self.designation = designation

    def generate_coordinates(self, num_points=100):
        m = int(self.designation[0]) / 100.0
        p = int(self.designation[1]) / 10.0
        t = int(self.designation[2:]) / 100.0

        x = np.linspace(0, 1, num_points)
        yc = np.where(x < p, m * (x / np.power(p, 2)) * (2 * p - x),
                      m * ((1 - x) / np.power(1 - p, 2)) * (1 + x - 2 * p))
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * np.power(x, 2) + 0.2843 * np.power(x, 3) - 0.1015 * np.power(x, 4))

        theta = np.arctan(np.gradient(yc, x))
        y_upper = yc + yt * np.cos(theta)
        y_lower = yc - yt * np.cos(theta)

        return AirfoilCoordinates(x, y_upper, y_lower)

class Transformation:
    """
    Base class for geometric transformations.
    """
    def apply(self, coordinates):
        # To be overridden in derived classes
        pass

class ScaleTransformation(Transformation):
    """
    Scale transformation for airfoil coordinates.
    """
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def apply(self, coordinates):
        x_scaled = coordinates.x * self.scale_factor
        y_upper_scaled = coordinates.y_upper * self.scale_factor
        y_lower_scaled = coordinates.y_lower * self.scale_factor
        return AirfoilCoordinates(x_scaled, y_upper_scaled, y_lower_scaled)

class InterpolateTransformation(Transformation):
    """
    Interpolate between two sets of airfoil coordinates.
    """
    def __init__(self, other_coordinates, interpolation_factor):
        self.other_coordinates = other_coordinates
        self.interpolation_factor = interpolation_factor

    def apply(self, coordinates):
        y_upper_interpolated = coordinates.y_upper * (1 - self.interpolation_factor) + self.other_coordinates.y_upper * self.interpolation_factor
        y_lower_interpolated = coordinates.y_lower * (1 - self.interpolation_factor) + self.other_coordinates.y_lower * self.interpolation_factor
        return AirfoilCoordinates(coordinates.x, y_upper_interpolated, y_lower_interpolated)

class WingSection:
    def __init__(self, root_airfoil, tip_airfoil, chord_length, interpolation_factor=0.5):
        self.root_airfoil = root_airfoil
        self.tip_airfoil = tip_airfoil
        self.chord_length = chord_length
        self.interpolation_factor = interpolation_factor

    def scaled_coordinates(self):
        root_coords = self.root_airfoil.generate_coordinates()
        tip_coords = self.tip_airfoil.generate_coordinates()

        # Normalize coordinates
        root_coords_normalized = root_coords.normalize()
        tip_coords_normalized = tip_coords.normalize()

        # Interpolate between root and tip airfoil shapes
        interpolation = InterpolateTransformation(tip_coords_normalized, self.interpolation_factor)
        interpolated_coords = root_coords_normalized.apply_transformation(interpolation)

        # Scale the interpolated coordinates to the current chord length
        scale = ScaleTransformation(self.chord_length)
        scaled_coords = interpolated_coords.apply_transformation(scale)

        return scaled_coords.x, scaled_coords.y_upper, scaled_coords.y_lower

def generate_triangles_between_airfoils(airfoil1, airfoil2, z1, z2):
    triangles = []
    for i in range(len(airfoil1) - 1):
        P1 = (airfoil1[i][0], airfoil1[i][1], z1)
        P2 = (airfoil1[i + 1][0], airfoil1[i + 1][1], z1)
        P3 = (airfoil2[i][0], airfoil2[i][1], z2)
        P4 = (airfoil2[i + 1][0], airfoil2[i + 1][1], z2)

        triangle1 = (P1, P2, P4)
        triangle2 = (P1, P4, P3)

        triangles.append(triangle1)
        triangles.append(triangle2)

    return triangles

class Wing:
    """
    Class to represent a wing with different airfoils at the root and tip.
    """
    def __init__(self, root_airfoil_designation, tip_airfoil_designation, span, root_chord, tip_chord, sweep_angle=0, dihedral_angle=0, num_sections=10):
        self.root_airfoil = Airfoil(root_airfoil_designation)
        self.tip_airfoil = Airfoil(tip_airfoil_designation)
        self.span = span
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.sweep_angle = sweep_angle
        self.dihedral_angle = dihedral_angle
        self.num_sections = num_sections
    
    def generate_mesh(self):
        wing_x, wing_y, wing_z_upper, wing_z_lower, wing_z = self.generate_raw_mesh_data()
        triangles = []
        for i in range(self.num_sections - 1):
            triangles += self._generate_section_triangles(i, wing_x, wing_y, wing_z_upper, wing_z_lower, wing_z)
        return triangles

    def _generate_section_triangles(self, section_index, wing_x, wing_y, wing_z_upper, wing_z_lower, wing_z):
        z1 = wing_z[section_index]
        z2 = wing_z[section_index + 1]
        upper_airfoil1 = np.vstack([wing_x[section_index], wing_z_upper[section_index]]).T
        upper_airfoil2 = np.vstack([wing_x[section_index + 1], wing_z_upper[section_index + 1]]).T
        lower_airfoil1 = np.vstack([wing_x[section_index], wing_z_lower[section_index]]).T
        lower_airfoil2 = np.vstack([wing_x[section_index + 1], wing_z_lower[section_index + 1]]).T

        upper_triangles = generate_triangles_between_airfoils(upper_airfoil1, upper_airfoil2, z1, z2)
        lower_triangles = generate_triangles_between_airfoils(lower_airfoil1, lower_airfoil2, z1, z2)
        return upper_triangles + lower_triangles

    
    
    def generate_raw_mesh_data(self):
        sweep_rad = np.radians(self.sweep_angle)
        dihedral_rad = np.radians(self.dihedral_angle)
        section_span = self.span / self.num_sections

        wing_x, wing_y, wing_z_upper, wing_z_lower = [], [], [], []

        for i in range(self.num_sections):
            spanwise_position = section_span * i
            sweep_offset = spanwise_position * np.tan(sweep_rad)
            dihedral_offset = spanwise_position * np.sin(dihedral_rad)
            chord_length = self.root_chord - (self.root_chord - self.tip_chord) * (spanwise_position / self.span)
            interpolation_factor = spanwise_position / self.span

            section = WingSection(self.root_airfoil, self.tip_airfoil, chord_length, interpolation_factor)
            x_scaled, y_upper_scaled, y_lower_scaled = section.scaled_coordinates()

            wing_x.append(x_scaled + sweep_offset)
            wing_y.append(np.full(len(x_scaled), spanwise_position))
            wing_z_upper.append(y_upper_scaled + dihedral_offset)
            wing_z_lower.append(y_lower_scaled + dihedral_offset)
            wing_z = np.linspace(0, self.span, self.num_sections)

        return wing_x, wing_y, wing_z_upper, wing_z_lower, wing_z
        
class WingMeshGenerator:
    def __init__(self, wing):
        self.wing = wing

    def generate_mesh(self):
        wing_x, wing_y, wing_z_upper, wing_z_lower = self.wing.generate_raw_mesh_data()
        # Further mesh processing can be added here
        return wing_x, wing_y, wing_z_upper, wing_z_lower

class WingConfiguration:
    """
    Class to handle the configuration of a wing from user input.
    """
    def __init__(self, config):
        self.tip_airfoil_designation = config.get('tip_airfoil_designation', '2412')
        self.root_airfoil_designation = config.get('root_airfoil_designation', '2412')
        self.span = config.get('span', 10)
        self.root_chord = config.get('root_chord', 1.5)
        self.tip_chord = config.get('tip_chord', 0.5)
        self.sweep_angle = config.get('sweep_angle', 30)
        self.dihedral_angle = config.get('dihedral_angle', 5)
        self.num_sections = config.get('num_sections', 10)

    def create_wing(self):
        return Wing(self.root_airfoil_designation, self.tip_airfoil_designation, self.span, self.root_chord, self.tip_chord, self.sweep_angle, self.dihedral_angle, self.num_sections)

class WingPlotter:
    def __init__(self, wing):
        self.wing = wing

    def plot_3d_mesh(self, view_angle=(30, -45)):
        triangles = self.wing.generate_mesh()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Convert triangles to a format suitable for Poly3DCollection
        tri_collection = Poly3DCollection(triangles, alpha=0.7)
        ax.add_collection3d(tri_collection)

        # Setting the limits based on the triangles
        all_points = np.array([point for triangle in triangles for point in triangle])
        ax.set_xlim(min(all_points[:,0]), max(all_points[:,0]))
        ax.set_ylim(min(all_points[:,1]), max(all_points[:,1]))
        ax.set_zlim(min(all_points[:,2]), max(all_points[:,2]))

        ax.view_init(*view_angle)
        ax.set_title('3D Wing Structure')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.show()


class STLExporter:
    def __init__(self, mesh_data):
        self.mesh_data = mesh_data

    @staticmethod
    def calculate_normal(triangle):
        # Calculate the normal for a triangle
        v1 = np.array(triangle[1]) - np.array(triangle[0])
        v2 = np.array(triangle[2]) - np.array(triangle[0])
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # Normalize the vector
        return normal

    def export_to_stl(self, file_name):
        with open(file_name, "w") as file:
            file.write("solid wing\n")
            for triangle in self.mesh_data:
                normal = self.calculate_normal(triangle)
                file.write(f"facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                file.write("    outer loop\n")
                for vertex in triangle:
                    file.write(f"        vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                file.write("    endloop\n")
                file.write("endfacet\n")
            file.write("endsolid wing\n")

    
# Example JSON input for user configuration
user_input_json = """
{
    "root_airfoil_designation": "2412",
    "tip_airfoil_designation": "0012",
    "span": 12,
    "root_chord": 1.8,
    "tip_chord": 0.6,
    "sweep_angle": 15,
    "dihedral_angle": 20,
    "num_sections": 5
}
"""

# Parse the JSON input and create the wing configuration
user_config = json.loads(user_input_json)
wing_config = WingConfiguration(user_config)
wing = wing_config.create_wing()
triangles = wing.generate_mesh()
exporter = STLExporter(triangles)
exporter.export_to_stl("wing_mesh.stl")

# Plotting the wing using WingPlotter
plotter = WingPlotter(wing)
plotter.plot_3d_mesh()  # Default isometric view
