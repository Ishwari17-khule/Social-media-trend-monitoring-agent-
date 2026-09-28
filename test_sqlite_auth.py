"""Test SQLite authentication setup."""
from database.auth_db_sqlite import AuthManager

print("Testing SQLite authentication...")

auth = AuthManager()

# Test database availability
if auth.is_available():
    print("SUCCESS: SQLite database is available")
else:
    print("ERROR: SQLite database is not available")
    exit(1)

# Test table creation
if auth.create_users_table():
    print("SUCCESS: Users table created successfully")
else:
    print("ERROR: Failed to create users table")
    exit(1)

# Test user verification with default admin
user = auth.verify_user("admin", "admin123")
if user:
    print(f"SUCCESS: Default admin user verified: {user}")
else:
    print("ERROR: Default admin user verification failed")
    exit(1)

# Test registration
success, message = auth.register_user("testuser", "testpass123")
if success:
    print(f"SUCCESS: Test user registration: {message}")
else:
    print(f"ERROR: Test user registration failed: {message}")
    exit(1)

# Test new user verification
user = auth.verify_user("testuser", "testpass123")
if user:
    print(f"SUCCESS: New test user verified: {user}")
else:
    print("ERROR: New test user verification failed")
    exit(1)

# Test wrong password
user = auth.verify_user("admin", "wrongpassword")
if user is None:
    print("SUCCESS: Wrong password correctly rejected")
else:
    print("ERROR: Wrong password should have been rejected")
    exit(1)

print("\nAll authentication tests passed successfully!")
print("Default credentials: admin / admin123")
