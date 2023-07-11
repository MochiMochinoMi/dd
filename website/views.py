from flask import Blueprint, render_template, request, session, redirect, url_for
import os
from .logic import extract_zip,plot_stock_data,plot_single_daily_data,avail_stocks,avail_stocksd
import glob
from flask import render_template, request, session, redirect, url_for
import time
from flask import redirect, url_for
from datetime import datetime



views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        uploaded_file = request.files['file']
        file_path = "uploaded_file.zip"
        uploaded_file.save(file_path)
        # Extract the ZIP file
        extracted_folder = extract_zip(file_path)
        # Construct the full folder path
        folder_path = os.path.join(extracted_folder)
        session['folder_path'] = folder_path
        print("Folder Path:", folder_path)
        return redirect(url_for('views.graph_choice'))
    return render_template('base.html')


@views.route('/graph_choice', methods=['GET', 'POST'])
def graph_choice():
    return render_template('graph_choice.html')




@views.route('/graph_selection', methods=['GET', 'POST'])
def graph_selection():
    if request.method == 'POST':
        if 'start_date' in request.form and 'end_date' in request.form:
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            session['start_date'] = start_date
            session['end_date'] = end_date
            # Check if stocks are available in the selected period
            folder_path = session.get('folder_path')
            stocks = avail_stocks(folder_path,start_date,end_date)
            session['stocks'] = stocks
            if stocks != None:
                return redirect(url_for('views.graph_details', stocks=stocks, show_form=True,start_date=start_date,end_date=end_date))
        # No stocks available in the selected period
            elif stocks == None:
                error = "No stocks available in the selected period."
                return render_template('graph_selection.html', error=error, show_form=True)
    # Render the initial form
    return render_template('graph_selection.html', show_form=True)



@views.route('/graph_details', methods=['GET', 'POST'])
def graph_details():
    stocks=session.get('stocks')
    start_date= session.get('start_date')
    end_date= session.get('end_date')
    folder_path = session.get('folder_path')
    if 'duration' in request.form and 'variable' in request.form and 'stocks' in request.form:
            duration = request.form.get('duration')
            variable = request.form.get('variable')
            selected_stocks = request.form.getlist('stocks')
            plot_stock_data(folder_path, start_date, end_date, duration, selected_stocks, variable,log_scale=False)
            plot_file_name = 'GRAPH.png'
            plot_file_path = os.path.join(plot_file_name)
            plot_file_nameqq = 'GRAPHss.png'
            if plot_file_path != None:
                return render_template('graph_details.html', show_form=True,plot_path=plot_file_path,plot_file_path=plot_file_path,stocks=stocks,plot_file_nameqq=plot_file_nameqq)
    # Render the initial form
    return render_template('graph_details.html', show_form=True,stocks=stocks)




@views.route('/graph_selectiond', methods=['GET', 'POST'])
def graph_selectiond():
    if request.method == 'POST':
        if 'selected_date' in request.form:
            selected_date = request.form.get('selected_date')
            session['selected_date'] = selected_date
            # Check if stocks are available in the selected period
            folder_path = session.get('folder_path')

            stocks = avail_stocksd(folder_path,selected_date)
            session['stocks'] = stocks
            if stocks != None:
                return redirect(url_for('views.graph_detailsd', stocks=stocks, show_form=True,selected_date=selected_date))
            elif stocks == None:
       # No stocks available in the selected period
                error = "No stocks available in the selected Date."
                return render_template('graph_selectiond.html', error=error, show_form=True)
    # Render the initial form
    return render_template('graph_selectiond.html', show_form=True)



@views.route('/graph_detailsd', methods=['GET', 'POST'])
def graph_detailsd():
    stocks=session.get('stocks')
    selected_date= session.get('selected_date')
    folder_path = session.get('folder_path')
    if 'stock' in request.form:
            selected_stocks = request.form.get('stock')
            log_scale=False
            log_scale = request.form.get('log_scale')
            plot_single_daily_data(folder_path, selected_stocks, selected_date, log_scale=log_scale)
            plot_file_name = 'GRAPH2.png'
            plot_file_path = os.path.join(plot_file_name)
            if plot_file_path != None:
                return render_template('graph_detailsd.html', show_form=True,plot_path=plot_file_path,plot_file_path=plot_file_path,stocks=stocks)
    # Render the initial form
    return render_template('graph_detailsd.html', show_form=True,stocks=stocks)























def delete_cached_images():
    static_folder = os.path.join(os.getcwd(),"website" ,'static')
    cached_images = glob.glob(os.path.join(static_folder, 'GRAPH*.png'))
    for image_path in cached_images:
        os.remove(image_path)
