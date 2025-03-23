CREATE TABLE IF NOT EXISTS reservations (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    hour VARCHAR(5) NOT NULL,
    reserved_by VARCHAR(255) DEFAULT 'Agent'
);

CREATE TABLE IF NOT EXISTS absences (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    date DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS user_threads (
    user_id VARCHAR(255) PRIMARY KEY,
    thread_id VARCHAR(255) NOT NULL
);
