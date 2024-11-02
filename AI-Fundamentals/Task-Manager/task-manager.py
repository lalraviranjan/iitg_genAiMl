# %%
# Imports
import getpass
import bcrypt
import pandas as pd
import sqlite3
from models.db_setup import create_connection

# %%
def connect_db():
    conn = create_connection()
    cursor = conn.cursor()
    return conn, cursor

# %%
# conn, cursor = connect_db()
# df = pd.read_sql_query("SELECT * from users", conn)
# print(df.to_string(index=False))

# %%
def is_user_registered(type, username):
    conn, cursor = connect_db()
    if type == 'login':
       cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    else:
        cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        
    result = cursor.fetchone()
    conn.close()
    if type == 'login':
        return (result[0], result[1]) if result else None
    else:
        return result if result else None

# %%
def register_user():
    conn, cursor = connect_db()
    while True:
        user_name = input("Enter username (or Type 'exit' to quit): ")
        if user_name.lower() == 'exit':
            conn.close()
            return "exit"
        
        if is_user_registered('register', user_name):
            print("\nUsername already exists. Please try a different username.")
            continue

        password = getpass.getpass("Enter password (or Type 'exit' to quit): ")
        if password.lower() == 'exit':
            conn.close()
            return "exit"
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (user_name, hashed_password))
            conn.commit()
            print("\nRegistration Successfull ! You can now log in to create your task.")
        except sqlite3.IntegrityError:
            print("\nUsername already exists. Please try different username.")
        finally:
            conn.close()

        return "registered"

# %%
def login():
    while True:
        name = input("Enter user name (or type 'exit' to quit): ")
        if name.lower() == 'exit':
            return "exit"
        
        user_data = is_user_registered('login', name)
        if user_data:
            password = getpass.getpass("Enter password (or type 'exit') to quit: ")
            if password.lower() == 'exit':
                return "exit"
            if bcrypt.checkpw(password.encode('utf-8'), user_data[1]):
                print(f"\nLogin successfull! Welcome to the Task Manager {name}.")
                return user_data[0]
            else:
                print("\nInavid Password. Please trye again.")
                continue
        else:
            print("\nUsername not found! Please register to login.")
            continue

# %%
def get_task(user_id):
    conn, cursor = connect_db()
    cursor.execute("SELECT task_id, description, status FROM tasks WHERE user_id = ?", (user_id,))
    result = cursor.fetchall()
    conn.close()

    return result if result else []

# %%
def view_task(user_id):
    task_data = get_task(user_id)
    print("*"*30)
    if task_data:
        df = pd.DataFrame(task_data, columns=["Task Id", "Description", "Status"])
        print(df.to_string(index=False))
    else:
        print("No Task added yet. Type 1 to Add your task.")

# %%
def add_task(user_id):
    description = input("Enter the task description (or type 'exit' to quit): ")
    if description.lower() == 'exit':
        return "exit"
    status = "Pending"

    conn, cursor = connect_db()
    try:
        cursor.execute("INSERT INTO tasks (user_id, description, status) VALUES (?, ?, ?)", (user_id, description, status))
        conn.commit()
        print("\nTask Added Successfully!")
    except sqlite3.Error as err:
        print("Error adding task: ", err)
    finally:
        conn.close()

# %%
def get_task_from_task_id(task_id):
    conn, cursor = connect_db()
    cursor.execute("SELECT 1 FROM tasks WHERE task_id = ?", (task_id,))
    result = cursor.fetchone()
    conn.close()
    return result if result else None

# %%
def update_task(task_id, type):
    conn, cursor = connect_db()
    if type == 'complete':
        cursor.execute("UPDATE tasks SET status = 'Completed' WHERE task_id = ?", (task_id,))
        conn.commit()
    elif type == 'delete':
       cursor.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
       conn.commit()
    else:
        print("Invalid type for task Update.")

# %%
def complete_task(user_id):
    while True:
        view_task(user_id)
        task_id = input("Enter the task id to mark as completed (or type 'exit' to quit): ")
        if task_id.lower() == 'exit':
            break
        task = get_task_from_task_id(task_id)
        if task:
            update_task(task_id, 'complete')
            print(f"Task Id - {task_id} marked as Completed.")
            break
        else:
            print("\nInvalid Task Id. Please enter the valid task id to mark as completed.")
            continue
    

# %%
def delete_task(user_id):
    while True:
        view_task(user_id)
        task_id = input("Enter the task id to delete (or type 'exit' to quit): ")
        if task_id.lower() == 'exit':
            return
        task = get_task_from_task_id(task_id)
        if task:
            while True:
                confirm_delete = input(f"Are you sure you want to delete Task id - {task_id} ? (type 'yes' to Confirm or 'exit' to quit): ")
                if confirm_delete.lower() == 'exit':
                    break
                elif confirm_delete.lower() == 'yes':
                    update_task(task_id, 'delete')
                    print(f"Task Id - {task_id} is deleted.")
                    break
                else:
                    print("\nPlease confirm a valid choice")
                    continue
            break
        else:
            print("\nInvalid Task Id. Please enter the valid task id to delete.")
            continue

# %%
def task_menu(user_id):
    while True:
        print("*"*30)
        print("Task Manager Menu: ")
        print("1. Add Task")
        print("2. View Task")
        print("3. Mark a Task as Completed")
        print("4. Delete a Task")
        print("5. Logout")

        option = input("Enter an option of your choice: ")
        if option == "1":
            add_task(user_id)
        elif option == "2":
            view_task(user_id)
        elif option == "3":
            complete_task(user_id)
        elif option == "4":
            delete_task(user_id)
        elif option == "5":
            return "logout"
        else:
            print("Please select a correct option")

# %%
def user_login_register():
    print("\nWelcome to Task Manager. Login to Manage your tasks.")
    print("*" * 30)
    while True:
        print("\nType 'L' to Login or 'R' to Register (or Type 'exit' to Exit)")
        user_input = input("Enter your choice: ").strip().upper()

        if user_input == 'L':
            user_id = login()
            if user_id == 'exit':
                continue
            
            login_status = task_menu(user_id)
            if login_status == 'logout':
                print("\nYou are logged out successfully!")
                break
        elif user_input == 'R':
            registration_status = register_user()
            if registration_status == 'exit':
                continue
        elif user_input.lower() == 'exit':
            print("Exiting the task manager.")
            break
        else:
            print("\nInvalid Choice. Please Type 'L' to Login or 'R' to Register (or Type 'E' to Exit)")

# %%
user_login_register()


