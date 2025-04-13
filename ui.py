import dearpygui.dearpygui as dpg

def build_ui(state):
    dpg.create_context()

    with dpg.window(label="Render Controls", width=300, height=400):
        dpg.add_checkbox(label="Denoising", default_value=state.denoising,
                         callback=lambda s: setattr(state, "denoising", dpg.get_value(s)))

        dpg.add_checkbox(label="Tone Mapping", default_value=state.tone_mapping,
                         callback=lambda s: setattr(state, "tone_mapping", dpg.get_value(s)))

        dpg.add_slider_int(label="SPP", default_value=state.spp, min_value=1, max_value=64,
                           callback=lambda s: setattr(state, "spp", dpg.get_value(s)))
        
        dpg.add_slider_int(label="Max Depth", default_value=state.max_depth, min_value=1, max_value=16,
                           callback=lambda s: setattr(state, "max_depth", dpg.get_value(s)))

        dpg.add_slider_int(label="Radius", default_value=state.radius, min_value=1, max_value=4,
                           callback=lambda s: setattr(state, "radius", dpg.get_value(s)))

        dpg.add_slider_float(label="Sigma Color", default_value=state.sigma_color, min_value=0.01, max_value=1.0,
                             callback=lambda s: setattr(state, "sigma_color", dpg.get_value(s)))

        dpg.add_slider_float(label="Sigma Normal", default_value=state.sigma_normal, min_value=0.01, max_value=1.0,
                             callback=lambda s: setattr(state, "sigma_normal", dpg.get_value(s)))

        dpg.add_slider_float(label="Sigma Spatial", default_value=state.sigma_spatial, min_value=0.1, max_value=5.0,
                             callback=lambda s: setattr(state, "sigma_spatial", dpg.get_value(s)))

        dpg.add_button(label="Reset Frame", callback=lambda: setattr(state, 'frame_count', 0))

    dpg.create_viewport(title='Renderer Settings', width=320, height=480)
    dpg.setup_dearpygui()
    dpg.show_viewport()
