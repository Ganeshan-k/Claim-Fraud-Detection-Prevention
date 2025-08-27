from app import db, User, EmailVerification, RecoveryCode

def confirm_clear():
    print("⚠️ This will delete all data from the database.")
    response = input("Proceed? (yes/no): ").strip().lower()
    return response == 'yes'

if __name__ == "__main__":
    try:
        from app import app
        with app.app_context():
            if confirm_clear():
                models = [User, EmailVerification, RecoveryCode]
                for model in models:
                    count = db.session.query(model).delete()
                    print(f"✅ Deleted {count} records from '{model.__tablename__}'")
                db.session.commit()
                print("🟢 All data cleared!")
            else:
                print("❌ Operation cancelled.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()