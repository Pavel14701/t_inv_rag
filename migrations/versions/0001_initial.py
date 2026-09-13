"""white API initial schema: candles, strategies, backtest_jobs, signals

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-13

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create white API tables."""
    op.create_table(
        "candles",
        sa.Column("inst_id", sa.String(length=64), nullable=False),
        sa.Column("ts", sa.BigInteger(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("inst_id", "ts"),
    )
    op.create_table(
        "strategies",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column("dsl_entry", sa.Text(), nullable=False),
        sa.Column("dsl_exit", sa.Text(), nullable=True),
        sa.Column("params_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_strategies_manifest", "strategies", ["manifest_hash"])
    op.create_table(
        "backtest_jobs",
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("request_json", sa.Text(), nullable=True),
        sa.Column("report_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("inst_id", sa.String(length=64), nullable=False),
        sa.Column("ts", sa.BigInteger(), nullable=False),
        sa.Column("direction", sa.String(length=8), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("sl_price", sa.Float(), nullable=False),
        sa.Column("tp_price", sa.Float(), nullable=False),
        sa.Column("p_win", sa.Float(), nullable=True),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_signals_inst_id_id", "signals", ["inst_id", "id"])


def downgrade() -> None:
    """Drop white API tables."""
    op.drop_index("ix_signals_inst_id_id", table_name="signals")
    op.drop_table("signals")
    op.drop_table("backtest_jobs")
    op.drop_index("ix_strategies_manifest", table_name="strategies")
    op.drop_table("strategies")
    op.drop_table("candles")
