"""add preset tables

Create Date: 2026-07-22 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import SqlPreset, SqlPresetVersion

# revision identifiers, used by Alembic.
revision = "e645f633357c"
down_revision = "a8b9c0d1e2f3"
branch_labels = None
depends_on = None


def upgrade():
    # Create the presets table (experiment_id, preset_name, preset_id)
    op.create_table(
        SqlPreset.__tablename__,
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("preset_name", sa.String(length=256), nullable=False),
        sa.Column("preset_id", sa.String(length=36), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_presets_experiment_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("preset_id", name="preset_pk"),
    )

    # Create the preset_versions table (preset_id, preset_version, serialized_preset, creation_time)
    op.create_table(
        SqlPresetVersion.__tablename__,
        sa.Column("preset_id", sa.String(length=36), nullable=False),
        sa.Column("preset_version", sa.Integer(), nullable=False),
        sa.Column("serialized_preset", sa.Text(), nullable=False),
        sa.Column("creation_time", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["preset_id"],
            ["presets.preset_id"],
            name="fk_preset_versions_preset_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("preset_id", "preset_version", name="preset_version_pk"),
    )

    # Create indexes
    with op.batch_alter_table(SqlPreset.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlPreset.__tablename__}_experiment_id_preset_name",
            ["experiment_id", "preset_name"],
            unique=True,
        )

    with op.batch_alter_table(SqlPresetVersion.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlPresetVersion.__tablename__}_preset_id",
            ["preset_id"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlPresetVersion.__tablename__)
    op.drop_table(SqlPreset.__tablename__)
