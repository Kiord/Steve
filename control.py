import taichi as ti
import numpy as np
from camera import compute_camera_vectors


class FreeFlyCameraController:
    def __init__(self, pos=np.array([2.0, 2.0, 2.0], dtype=np.float32),
                 yaw=-135.0, pitch=-45.0, move_speed=1.0, turn_speed=90.0,
                 fov=60):
        self.pos = np.asanyarray(pos)
        self.yaw = yaw
        self.pitch = pitch
        self.move_speed = move_speed
        self.turn_speed = turn_speed
        self.fov = fov

    def get_view_dirs(self):
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)

        forward = np.array([
            np.cos(pitch_rad) * np.cos(yaw_rad),
            np.sin(pitch_rad),
            np.cos(pitch_rad) * np.sin(yaw_rad)
        ], dtype=np.float32)
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        return forward, right, up

    def update_from_input(self, gui :ti.ui.Window, dt):
        forward, right, _ = self.get_view_dirs()

        # for e in gui.get_events(ti.ui.PRESS):
        #     print(f"Pressed key: {e.key}")

        if gui.is_pressed("w"):  # Forward
            self.pos += forward * self.move_speed * dt
        if gui.is_pressed("s"):  # Backward
            self.pos -= forward * self.move_speed * dt
        if gui.is_pressed("a"):  # Left
            self.pos -= right * self.move_speed * dt
        if gui.is_pressed("d"):  # Right
            self.pos += right * self.move_speed * dt
        if gui.is_pressed(" "):  # Up
            self.pos += np.array([0, 1, 0], dtype=np.float32) * self.move_speed * dt
        if gui.is_pressed('Control'):  # Down
            self.pos -= np.array([0, 1, 0], dtype=np.float32) * self.move_speed * dt

        if gui.is_pressed("Left"):  # Yaw left
            self.yaw -= self.turn_speed * dt
        if gui.is_pressed("Right"):  # Yaw right
            self.yaw += self.turn_speed * dt
        if gui.is_pressed("Up"):  # Pitch up
            self.pitch = min(self.pitch + self.turn_speed * dt, 89.0)
        if gui.is_pressed("Down"):  # Pitch down
            self.pitch = max(self.pitch - self.turn_speed * dt, -89.0)

    def update_camera_field(self, camera_field, aspect_ratio):
        forward, _, up = self.get_view_dirs()
        lookat = self.pos + forward
        lower_left_corner, horizontal, vertical = compute_camera_vectors(
            self.pos, lookat, up, self.fov, aspect_ratio)

        camera_field[None].origin = ti.Vector(list(self.pos))
        camera_field[None].lower_left_corner = ti.Vector(list(lower_left_corner))
        camera_field[None].horizontal = ti.Vector(list(horizontal))
        camera_field[None].vertical = ti.Vector(list(vertical))