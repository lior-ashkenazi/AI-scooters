import tkinter
from tkintermapview import TkinterMapView
from tkintermapview.canvas_path import CanvasPath
from tkintermapview.canvas_position_marker import CanvasPositionMarker


class CustomCanvasPath(CanvasPath):

    def draw(self, move=False):
        new_line_length = self.last_position_list_length != len(self.position_list)
        self.last_position_list_length = len(self.position_list)

        widget_tile_width = self.map_widget.lower_right_tile_pos[0] - self.map_widget.upper_left_tile_pos[0]
        widget_tile_height = self.map_widget.lower_right_tile_pos[1] - self.map_widget.upper_left_tile_pos[1]

        if move is True and self.last_upper_left_tile_pos is not None and new_line_length is False:
            x_move = ((self.last_upper_left_tile_pos[0] - self.map_widget.upper_left_tile_pos[0]) / widget_tile_width) * self.map_widget.width
            y_move = ((self.last_upper_left_tile_pos[1] - self.map_widget.upper_left_tile_pos[1]) / widget_tile_height) * self.map_widget.height

            for i in range(0, len(self.position_list)* 2, 2):
                self.canvas_line_positions[i] += x_move
                self.canvas_line_positions[i + 1] += y_move
        else:
            self.canvas_line_positions = []
            for position in self.position_list:
                canvas_position = self.get_canvas_pos(position, widget_tile_width, widget_tile_height)
                self.canvas_line_positions.append(canvas_position[0])
                self.canvas_line_positions.append(canvas_position[1])

        if not self.deleted:
            if self.canvas_line is None:
                self.map_widget.canvas.delete(self.canvas_line)
                self.canvas_line = self.map_widget.canvas.create_line(self.canvas_line_positions,
                                                                      width=3, fill=self.path_color,
                                                                      capstyle=tkinter.ROUND, joinstyle=tkinter.ROUND,
                                                                      tag="path", arrow=tkinter.LAST)

                if self.command is not None:
                    self.map_widget.canvas.tag_bind(self.canvas_line, "<Enter>", self.mouse_enter)
                    self.map_widget.canvas.tag_bind(self.canvas_line, "<Leave>", self.mouse_leave)
                    self.map_widget.canvas.tag_bind(self.canvas_line, "<Button-1>", self.click)
            else:
                self.map_widget.canvas.coords(self.canvas_line, self.canvas_line_positions)
        else:
            self.map_widget.canvas.delete(self.canvas_line)
            self.canvas_line = None

        self.map_widget.manage_z_order()
        self.last_upper_left_tile_pos = self.map_widget.upper_left_tile_pos

class CustomCanvasPositionMarker(CanvasPositionMarker):
    def draw(self, event=None):
        canvas_pos_x, canvas_pos_y = self.get_canvas_pos(self.position)

        if not self.deleted:
            if 0 - 50 < canvas_pos_x < self.map_widget.width + 50 and 0 < canvas_pos_y < self.map_widget.height + 70:
                if self.polygon is None:
                    self.polygon = self.map_widget.canvas.create_polygon(canvas_pos_x - 14, canvas_pos_y - 23,
                                                                         canvas_pos_x, canvas_pos_y,
                                                                         canvas_pos_x + 14, canvas_pos_y - 23,
                                                                         fill=self.marker_color_outside, width=2,
                                                                         outline=self.marker_color_outside,
                                                                         tag="marker")
                    if self.command is not None:
                        self.map_widget.canvas.tag_bind(self.polygon, "<Enter>", self.mouse_enter)
                        self.map_widget.canvas.tag_bind(self.polygon, "<Leave>", self.mouse_leave)
                        self.map_widget.canvas.tag_bind(self.polygon, "<Button-1>", self.click)
                else:
                    self.map_widget.canvas.coords(self.polygon,
                                                  canvas_pos_x - 14, canvas_pos_y - 23,
                                                  canvas_pos_x, canvas_pos_y,
                                                  canvas_pos_x + 14, canvas_pos_y - 23)
                if self.big_circle is None:
                    self.big_circle = self.map_widget.canvas.create_oval(canvas_pos_x - 14, canvas_pos_y - 45,
                                                                         canvas_pos_x + 14, canvas_pos_y - 17,
                                                                         fill=self.marker_color_circle, width=6,
                                                                         outline=self.marker_color_outside,
                                                                         tag="marker")
                    if self.command is not None:
                        self.map_widget.canvas.tag_bind(self.big_circle, "<Enter>", self.mouse_enter)
                        self.map_widget.canvas.tag_bind(self.big_circle, "<Leave>", self.mouse_leave)
                        self.map_widget.canvas.tag_bind(self.big_circle, "<Button-1>", self.click)
                else:
                    self.map_widget.canvas.coords(self.big_circle,
                                                  canvas_pos_x - 14, canvas_pos_y - 45,
                                                  canvas_pos_x + 14, canvas_pos_y - 17)

                if self.text is not None:
                    if self.canvas_text is None:
                        self.canvas_text = self.map_widget.canvas.create_text(canvas_pos_x, canvas_pos_y - 56,
                                                                              anchor=tkinter.S,
                                                                              text=self.text,
                                                                              fill=self.text_color,
                                                                              font=self.font,
                                                                              tag=("marker", "marker_text"))
                        if self.command is not None:
                            self.map_widget.canvas.tag_bind(self.canvas_text, "<Enter>", self.mouse_enter)
                            self.map_widget.canvas.tag_bind(self.canvas_text, "<Leave>", self.mouse_leave)
                            self.map_widget.canvas.tag_bind(self.canvas_text, "<Button-1>", self.click)
                    else:
                        self.map_widget.canvas.coords(self.canvas_text, canvas_pos_x, canvas_pos_y - 56)
                        self.map_widget.canvas.itemconfig(self.canvas_text, text=self.text)
                else:
                    if self.canvas_text is not None:
                        self.map_widget.canvas.delete(self.canvas_text)

                if self.image is not None and self.image_zoom_visibility[0] <= self.map_widget.zoom <= \
                        self.image_zoom_visibility[1] \
                        and not self.image_hidden:

                    if self.canvas_image is None:
                        self.canvas_image = self.map_widget.canvas.create_image(canvas_pos_x, canvas_pos_y - 85,
                                                                                anchor=tkinter.S,
                                                                                image=self.image,
                                                                                tag=("marker", "marker_image"))
                    else:
                        self.map_widget.canvas.coords(self.canvas_image, canvas_pos_x, canvas_pos_y - 85)
                else:
                    if self.canvas_image is not None:
                        self.map_widget.canvas.delete(self.canvas_image)
                        self.canvas_image = None
            else:
                self.map_widget.canvas.delete(self.polygon, self.big_circle, self.canvas_text, self.canvas_image)
                self.polygon, self.big_circle, self.canvas_text, self.canvas_image = None, None, None, None

            self.map_widget.manage_z_order()

class CustomTkinterMapView(TkinterMapView):
    def set_path(self, position_list: list, **kwargs) -> CanvasPath:
        path = CustomCanvasPath(self, position_list, **kwargs)
        path.draw()
        self.canvas_path_list.append(path)
        return path

    def set_marker(self, deg_x: float, deg_y: float, text: str = None, **kwargs) -> CanvasPositionMarker:
        marker = CustomCanvasPositionMarker(self, (deg_x, deg_y), text=text, **kwargs)
        marker.draw()
        self.canvas_marker_list.append(marker)
        return marker
