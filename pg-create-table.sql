CREATE TYPE label_type AS ENUM (
  'a',
  'h',
  't',
  'v'
);

CREATE TABLE results (
  run INTEGER NOT NULL,
  model INTEGER NOT NULL,
  label label_type NOT NULL,
  flags INTEGER NOT NULL,
  iter INTEGER NOT NULL,
  train_acc DOUBLE PRECISION NOT NULL,
  train_loss DOUBLE PRECISION NOT NULL,
  valid_acc DOUBLE PRECISION NOT NULL,
  valid_loss DOUBLE PRECISION NOT NULL,
  valid_precision DOUBLE PRECISION NOT NULL,
  valid_recall DOUBLE PRECISION NOT NULL,
  valid_f10 DOUBLE PRECISION NOT NULL,
  valid_f05 DOUBLE PRECISION NOT NULL,
  valid_tp INTEGER NOT NULL,
  valid_fp INTEGER NOT NULL,
  valid_fn INTEGER NOT NULL,
  valid_tn INTEGER NOT NULL,
  test_acc DOUBLE PRECISION NOT NULL,
  test_loss DOUBLE PRECISION NOT NULL,
  test_precision DOUBLE PRECISION NOT NULL,
  test_recall DOUBLE PRECISION NOT NULL,
  test_f10 DOUBLE PRECISION NOT NULL,
  test_f05 DOUBLE PRECISION NOT NULL,
  test_tp INTEGER NOT NULL,
  test_fp INTEGER NOT NULL,
  test_fn INTEGER NOT NULL,
  test_tn INTEGER NOT NULL,
  PRIMARY KEY (run, model, label, flags, iter)
);

CREATE INDEX ON results (model);
CREATE INDEX ON results (label);
CREATE INDEX ON results (flags);
CREATE INDEX ON results (valid_acc);
CREATE INDEX ON results (valid_loss);
CREATE INDEX ON results (valid_precision);
