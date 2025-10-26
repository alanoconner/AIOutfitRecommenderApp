# import tkinter as tk
# from tkinter import filedialog
# from tkinter import messagebox
# from PIL import Image, ImageTk
# import requests
# from io import BytesIO
# import repo as rp
#
#
# def upload_picture():
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         picture_label.config(text=f"Selected: {file_path.split('/')[-1]}")
#         picture_label.image_path = file_path
#
# def show_image(image_url):
#     try:
#         response = requests.get(image_url)
#         response.raise_for_status()
#         img_data = response.content
#
#         img = Image.open(BytesIO(img_data))
#
#         img_window = tk.Toplevel()
#         img_window.title("Downloaded Image")
#
#         photo = ImageTk.PhotoImage(img)
#
#         label = tk.Label(img_window, image=photo)
#         label.image = photo  # Keep a reference to avoid garbage collection
#         label.pack()
#
#     except requests.RequestException as e:
#         messagebox.showerror("Error", f"Failed to download image: {e}")
#     except Exception as e:
#         messagebox.showerror("Error", f"Failed to display image: {e}")
#
# def submit():
#     gender = gender_var.get()
#     height = height_entry.get()
#     image_path = getattr(picture_label, 'image_path', None)
#     if not gender or not height:
#         messagebox.showwarning("Input Error", "Please provide both gender and height")
#     else:
#
#         image_link = rp.generator(image_path, gender, height)
#         # messagebox.showinfo("Success", f"Gender: {gender}\nHeight: {height}\nPicture: {}")
#         show_image(image_link)
#
#
#
#
# # Create main window
# root = tk.Tk()
# root.title("research_project_v0.1")
#
# # Gender selection
# gender_var = tk.StringVar(value="Select Gender")
# tk.Label(root, text="Gender:").grid(row=0, column=0, padx=10, pady=10)
# tk.OptionMenu(root, gender_var, "Male", "Female", "Other").grid(row=0, column=1, padx=10, pady=10)
#
# # Height entry
# tk.Label(root, text="Height (cm):").grid(row=1, column=0, padx=10, pady=10)
# height_entry = tk.Entry(root)
# height_entry.grid(row=1, column=1, padx=10, pady=10)
#
# # Picture upload
# tk.Label(root, text="Picture:").grid(row=2, column=0, padx=10, pady=10)
# picture_label = tk.Label(root, text="No file selected")
# picture_label.grid(row=2, column=1, padx=10, pady=10)
# tk.Button(root, text="Upload Picture", command=upload_picture).grid(row=2, column=2, padx=10, pady=10)
#
# # Submit button
# tk.Button(root, text="Submit", command=submit).grid(row=3, column=0, columnspan=3, padx=10, pady=20)
#
# # Run the application
# root.mainloop()
