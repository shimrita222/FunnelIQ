create table customers (
  id bigint generated always as identity primary key,
  ad_budget numeric not null,
  num_leads integer not null,
  leads_answered integer not null,
  leads_not_answered integer not null,
  followup_1 integer not null,
  followup_2 integer not null,
  followup_3 integer not null,
  followup_4 integer not null,
  followup_5 integer not null,
  not_closed integer not null,
  closed integer not null,
  calls_to_closed numeric not null,
  calls_to_not_closed numeric not null,
  customer_acquisition_cost numeric not null,
  ltv_months numeric,
  purchased boolean not null,
  upsell boolean not null,
  cumulative_profit numeric,
  referred boolean not null,
  created_at timestamptz not null default now()
);

alter table customers enable row level security;

create policy "Authenticated users can read customers"
  on customers for select
  to authenticated
  using (true);
