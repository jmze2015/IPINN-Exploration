import os
import pandas as pd 
import plotly.express as px
import imageio.v2 as imageio

## idea function: csv -> figure
## how to read e.g. pd.read_csv("Data/lin-adv-2.csv")

"""
Below are two animation functions for two different purposes:
    >> anim_plot_post is a function intended for use after NN model predictions have been created and stored. 
       This first function plots both the exact solution and the NN predictions at the same time and allows us to 
       visualize the error with time evolution.

    >> anim_plot_pre is a function intended for pre NN model prediction visualization. That is we can visualize a 
       a single numerical solution under time evolution.

    >> We perhaps need a third function to visualize 3 numerical solutions at once: i.e. something like LF, PINN, and IPINNs 
       all at once.
"""

def anim_plot_post(filename, main, xlim, ylim):
    data = pd.read_csv(filename)

    fig = px.line(
        data, 
        x = "x", 
        y = ["u", "u_pred"],
        animation_frame="time_step",
        title=main
    )

    fig.update_layout(
        xaxis = dict(title="x", range = xlim),
        yaxis = dict(title="u(x,t)", range = ylim)
    )

    fig.update_layout(
        sliders=[{
            "currentvalue" : {
                "prefix" : "t = "
            }
        }]
    )

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 20

    fig.show()

def anim_plot_pre(filename, main, xlim, ylim):
    data = pd.read_csv(filename)

    fig = px.line(
        data, 
        x = "x", 
        y = "u",
        animation_frame="t",
        title=main
    )

    fig.update_layout(
        xaxis = dict(title="x", range = xlim),
        yaxis = dict(title="u(x,t)", range = ylim)
    )

    fig.update_layout(
        sliders=[{
            "currentvalue" : {
                "prefix" : "t = "
            }
        }]
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50

    fig.show()

"""
Examples below of anim_plot_pre and anim_plot_post:

## anim_plot_pre("Data/burgers_lf_tmax_2.csv", main = "GGs", xlim = [-1.2,5.2], ylim=[-0.2, 1.2])
## anim_plot_post("Results/burgers1-predictions.csv", main="Burgers: Exact vs NN (100 Epochs)", xlim=[-5,5], ylim=[-0.2,3])

Functions are meant to animate the time evolution of the numerical solutions. anim_plot_pre is to animate the numerical solutions or a method, 
while anim_plot_post assumes a NN with columns for both numerical solutions and a prediction column. 

Code could further be generalized for comparison of two numerical/analytical/mixed solutions.
"""

#anim_plot_post("Results/NN_literature_1000_epochs.csv", main="Literature NN (1000 Epochs)", xlim = [-10,10], ylim=[-4, 2.2])
#anim_plot_pre("Data/M_500_literature_data.csv", main="Initial Data (from literature)", xlim=[-10, 10], ylim=[-4,3])


"""
Function below is used to create mp4 files of the animation
"""

def csv_to_mp4_post(filename, main, xlim, ylim, output_name=None, fps=30, step=1):
    """
    Converts a results CSV into an MP4 animation.

    Assumes CSV has columns:
        x, time_step, u, u_pred

    Parameters:
        filename: path to CSV file
        main: title of plot
        xlim: list/tuple, e.g. [-2.5, 2.5]
        ylim: list/tuple, e.g. [-0.2, 1.2]
        output_name: name of mp4 file; if None, uses CSV name
        fps: frames per second
        step: keep every `step`-th time step
    """

    data = pd.read_csv(filename)

    os.makedirs("MP4", exist_ok=True)
    os.makedirs("temp_frames", exist_ok=True)

    if output_name is None:
        base = os.path.splitext(os.path.basename(filename))[0]
        output_name = base + ".mp4"

    output_path = os.path.join("MP4", output_name)

    time_steps = sorted(data["time_step"].unique())
    time_steps = time_steps[::step]

    frame_files = []

    for i, n in enumerate(time_steps):
        frame_data = data[data["time_step"] == n]

        fig = px.line(
            frame_data,
            x="x",
            y=["u", "u_pred"],
            title=f"{main}<br>time step = {n}"
        )

        fig.update_layout(
            xaxis=dict(title="x", range=xlim),
            yaxis=dict(title="u(x,t)", range=ylim),
            width=900,
            height=600
        )

        frame_file = f"temp_frames/frame_{i:04d}.png"
        fig.write_image(frame_file)
        frame_files.append(frame_file)

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)

    for frame_file in frame_files:
        os.remove(frame_file)

    print(f"Saved MP4 to: {output_path}")


"""
Function below is used to create mp4 files of the animation
"""

# csv_to_mp4_post(filename="Results/burgers1-predictions.csv", main="Burgers': Numerical vs NN (100 Epochs)", xlim=[-2.5,3.0], ylim=[-0.2, 1.2],
#                 output_name="burgers_NN_100epochs.mp4", fps=30)







def save_anim_post(filename, main, xlim, ylim, output_name):
    data = pd.read_csv(filename)

    fig = px.line(
        data,
        x="x",
        y=["u", "u_pred"],
        animation_frame="time_step",
        title=main
    )

    fig.update_layout(
        xaxis=dict(title="x", range=xlim),
        yaxis=dict(title="u(x,t)", range=ylim)
    )

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 20

    os.makedirs("Animations", exist_ok=True)

    fig.write_html(
        f"Animations/{output_name}.html",
        include_plotlyjs="cdn"
    )


def save_anim_pre(filename, main, xlim, ylim, output_name):
    data = pd.read_csv(filename)

    fig = px.line(
        data,
        x="x",
        y="u",
        animation_frame="t",
        title=main
    )

    fig.update_layout(
        xaxis=dict(title="x", range=xlim),
        yaxis=dict(title="u(x,t)", range=ylim)
    )

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50

    os.makedirs("Animations", exist_ok=True)

    fig.write_html(
        f"Animations/{output_name}.html",
        include_plotlyjs="cdn"
    )

#save_anim_pre("Data/M_500_literature_data.csv", main = "Numerical Solution of Proposed Initial Data (Goduvnov)", xlim=[-10, 10], ylim=[-4.2,2], output_name="M_500_literature_data")
#save_anim_post("Results/PINN_literature_5000epochs.csv", main = "Numerical Solution vs PINN Prediction (5000 Epochs)", xlim=[-10, 10], ylim=[-4.2,2], output_name="PINN_literature_5000epochs")


