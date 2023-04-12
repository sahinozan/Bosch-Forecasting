from typing import Any
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import openpyxl
from openpyxl import styles
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def find_common_pipes(file_index: int,
                      file_year: int,
                      top_level_df: pd.DataFrame,
                      file_dict: dict[int, list[str]],
                      master_dir: str
                      ) -> pd.DataFrame:
    """
    Finds the top 20 produced pipes for the selected production plan

    Args:
        file_index: The file index (4, 9, 27, 36, 41,...)
        file_year: The file year (2021, 2022, 2023)
        top_level_df: The top level dataframe
        file_dict: The file dictionary
        master_dir: The master directory

    Returns:
        The top level dataframe with the top 20 pipes for the selected production plan
    """
    df = pd.read_excel(f'{master_dir}/{str(file_year)}/{file_dict[file_year][file_index]}', sheet_name='Pivot')

    # add the total quantity column
    df['Total'] = df.iloc[:, 5:26].sum(axis=1)
    df.loc["Hat", "Total"] = 0

    # sort the dataframe by the Total column
    df.sort_values(by=['Total'], inplace=True, ascending=False)

    # transpose the dataframe and select the top 20 pipes
    top_20_pipes = df.iloc[:20, [1, -1]].T

    # insert an empty column to first position
    top_20_pipes.insert(0, "X", file_dict[file_year][file_index].split("_")[0] + f"_{file_year}")

    # rename the columns
    top_20_pipes.columns = ["X", *range(1, 21)]
    top_20_pipes.index = pd.Index(['Pipe TTNr', 'Total'])

    # add the top 20 pipes to the top_level_df
    top_level_df = pd.concat([top_level_df, top_20_pipes], axis=0)
    return top_level_df


def configure_matplotlib(labelsize: int = 18,
                         titlesize: int = 22,
                         titlepad: int = 25,
                         labelpad: int = 15,
                         dpi: int = 200) -> None:
    """
    Configures matplotlib to use the fivethirtyeight style and the Ubuntu font.
    Args:
        labelsize: The size of the axis labels
        titlesize: The size of the title
        titlepad: The padding of the title
        labelpad: The padding of the axis labels
        dpi: The resolution of the figure
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = labelsize
    plt.rcParams['axes.labelpad'] = labelpad
    plt.rcParams['axes.titlesize'] = titlesize
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = titlepad
    plt.rcParams['figure.dpi'] = dpi

    # change the background color of the figure
    plt.rcParams['figure.facecolor'] = 'none'
    plt.rcParams['axes.facecolor'] = 'none'

    # change the color of the grid
    plt.rcParams['axes.grid'] = False


def hex_to_RGB(hex_str) -> list[int]:
    """
    Converts a hex string to RGB. (FFFFFF -> [255,255,255])
    Args:
        hex_str: The hex string

    Returns:
        A list of RGB values
    """
    return [int(hex_str[x:x + 2], 16) for x in range(1, 6, 2)]


def get_color_gradient(c1, c2, n) -> list[str]:
    """
    Given two hex colors, returns a color gradient
    with n colors.

    Args:
        c1: The first color (hex)
        c2: The second color (hex)
        n: The number of colors in the gradient

    Returns:
        A list of hex colors
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val * 255)), "02x") for val in item]) for item in rgb_colors]


def create_bar_plot(df: pd.DataFrame, selected_year: int, file_index: str) -> None:
    """
    Creates a bar plot for the selected year and file index

    Args:
        df: The dataframe that contains the data (exp_df)
        selected_year: The selected year (2021, 2022, 2023)
        file_index: The file index (4, 9, 27, 36, 41,...)
    """
    if len(str(file_index)) == 1:
        file_index = "0" + str(file_index)

    selected_file = f'KW{str(file_index)}_{selected_year}'

    if (str(selected_year), selected_file, "Pipe TTNr") not in df.columns:
        print(f">>> KW{str(file_index)} does not exist for the year {selected_year}!")
        return None

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(20, 10))

    # sorted plot
    sns.barplot(ax=ax,
                data=df,
                x=df[(str(selected_year), selected_file, "Pipe TTNr")],
                y=df[(str(selected_year), selected_file, "Total")],
                order=df.sort_values(by=(str(selected_year), selected_file, "Total"), ascending=True).head(20)[
                    (str(selected_year), selected_file, "Pipe TTNr")],
                color="blue",
                label=selected_file,
                errorbar=None,
                palette=get_color_gradient("#1f77b4", "#ff7f0e", 20))

    # disable outline of the bars
    # sns.despine(fig=fig, ax=ax, top=False, right=False, left=False, bottom=False, offset=False, trim=False)

    sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)

    # rotate the x-ticks
    plt.xticks(rotation=90)

    # add padding to x-ticks
    plt.tick_params(axis='x', which='major', pad=10)

    plt.xlabel("Pipe TTNr", labelpad=25)
    plt.ylabel("Total Quantity", labelpad=25)
    plt.title(f"Top 20 Pipes in {selected_file}", pad=25)

    plt.show()


def unique_pipe_bar_plot(final_df: pd.DataFrame) -> None:
    """
    Creates a bar plot for the unique pipes
    Args:
        final_df: The dataframe that contains the data (final_df)
    """
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(15, 20))

    # sorted plot
    ax = sns.barplot(y="Pipe TTNr",
                     x="Total",
                     data=final_df.loc[final_df["Pipe TTNr"].apply(lambda x: x.isnumeric()), :].copy())

    # add grid lines
    ax.grid(axis="x", color="black", linestyle="dashed", linewidth=0.5)

    sns.despine(fig=fig, ax=ax, top=True, right=True, left=True, bottom=True)

    # rotate the x-ticks
    plt.xticks(rotation=90)

    # add padding to x-ticks
    plt.tick_params(axis='y', which='major', pad=10)

    plt.ylabel("Pipe TTNr", labelpad=25)
    plt.xlabel("Total Quantity", labelpad=25)

    plt.show()


def format_general_sheet(file_dir: str) -> None:
    """
    Format the Excel file. This formatting involves removing the blank rows, setting the column header,
    setting the column, row width and height and, setting the font size and alignment.

    Args:
        file_dir: The file directory of the Excel file
    """
    wb = openpyxl.load_workbook(file_dir)
    if "General" in wb.sheetnames:
        ws = wb["General"]
    else:
        print(">>> The sheet 'General' does not exist!")
        return None

    # remove the blank rows
    ws.delete_rows(3)

    # set the column header
    ws['A2'] = "Ranking"

    # set the column width
    for i in range(1, 194):
        if i % 2 == 0:
            ws.column_dimensions[get_column_letter(i)].width = 15  # type: ignore
        elif i % 2 == 1:
            ws.column_dimensions[get_column_letter(i)].width = 10  # type: ignore

    # set the row height
    for i in range(1, 23):
        ws.row_dimensions[i].height = 20  # type: ignore
        openpyxl.worksheet.dimensions.Dimensions(row=i, height=20)  # type: ignore

    # set the font size and alignment
    for i in range(1, 23):
        for j in range(1, 194):
            ws.cell(row=i, column=j).font = Font(size=10)
            ws.cell(row=i, column=j).alignment = Alignment(horizontal='center', vertical='center')

    # set the column header font size and font bold
    for i in range(1, 194):
        ws.cell(row=1, column=i).font = Font(size=12, bold=True)
        ws.cell(row=2, column=i).font = Font(size=10, bold=True)

    for i in range(1, 23):
        ws.cell(row=i, column=1).font = Font(size=10, bold=True)

    # create the borders
    for i in range(1, 23):
        for j in range(1, 194):
            ws.cell(row=i, column=j).border = Border(
                left=styles.borders.Side(border_style='thin', color='000000'),
                right=styles.borders.Side(border_style='thin',
                                          color='000000'),
                top=styles.borders.Side(border_style='thin', color='000000'),
                bottom=styles.borders.Side(border_style='thin',
                                           color='000000'))

    wb.save(file_dir)


def format_experimental_sheet(file_dir: str) -> None:
    """
    Format the Excel file. This formatting involves removing the blank rows, setting the column header,
    setting the column, row width and height and, setting the font size and alignment.

    Args:
        file_dir: The file directory of the Excel file
    """
    wb = openpyxl.load_workbook(file_dir)
    if "Experimental" in wb.sheetnames:
        ws = wb["Experimental"]
    else:
        print(">>> The sheet 'Experimental' does not exist!")
        return None

    # set the column header
    ws['A1'] = "File Index"
    ws['B1'] = "Ranking"

    # set the column width
    for i in range(2, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(i)].width = 15  # type: ignore

    # set the column of the file index column
    ws.column_dimensions[get_column_letter(1)].width = 20  # type: ignore

    # set the row height
    for i in range(1, ws.max_row + 1):
        ws.row_dimensions[i].height = 20  # type: ignore

    # set the font size and alignment
    for i in range(1, ws.max_row + 1):
        for j in range(1, ws.max_column + 1):
            ws.cell(row=i, column=j).font = Font(size=10)
            ws.cell(row=i, column=j).alignment = Alignment(horizontal='center', vertical='center')

    # set the column header font size and font bold
    for i in range(1, ws.max_column + 1):
        ws.cell(row=1, column=i).font = Font(size=10, bold=True)

    for i in range(1, ws.max_row + 1):
        ws.cell(row=i, column=1).font = Font(size=12, bold=True)
        ws.cell(row=i, column=2).font = Font(size=10, bold=True)

    # create the borders
    for i in range(1, ws.max_row + 1):
        for j in range(1, ws.max_column + 1):
            ws.cell(row=i, column=j).border = Border(
                left=Side(border_style='thin', color='000000'),
                right=Side(border_style='thin',
                           color='000000'),
                top=Side(border_style='thin', color='000000'),
                bottom=Side(border_style='thin',
                            color='000000'))

    wb.save(file_dir)


def create_three_level_index(df: pd.DataFrame) -> list[tuple[str | Any, ...]]:
    """
    Creates a three level index for the dataframe

    Args:
        df: The dataframe that contains the data (exp_df)

    Returns:
        The three level index
    """
    final_columns = []

    for v, i in enumerate(df.columns):
        if i[0].split("_")[-1] == "2021":
            w = list(i)
            w.insert(0, "2021")
        elif i[0].split("_")[-1] == "2022":
            w = list(i)
            w.insert(0, "2022")
        else:
            w = list(i)
            w.insert(0, "2023")
        final_columns.append(tuple(w))

    return final_columns


def get_data_files() -> tuple[dict[int, list[str]], str, str]:
    """
    Gets the data files and the file directory

    Returns:
        The data files, the file directory and the master directory
    """
    file_dict = {}

    file_dir = "/Users/ozansahin/Downloads/master.xlsx"
    master_dir = "/Users/ozansahin/Downloads/New_Converted_Data"

    for i in [2021, 2022, 2023]:
        os.chdir(f"{master_dir}/{i}")
        extension = 'xlsx'
        file_dict[i] = [i for i in glob.glob('*.{}'.format(extension))]

    return file_dict, file_dir, master_dir
